# Improvement Recommendations for CyclistPrediction

## TL;DR

- **Implicit gesture classifier is near-broken**: Left_Look at 25.9% accuracy (worse than random for 4 classes). Fix with class-weighted loss, data augmentation, and better architecture.
- **Replace Bi-LSTM with ST-GCN or Transformer**: vanilla LSTM ignores skeleton graph topology. ST-GCN/CTR-GCN model joint connectivity directly and are SOTA for skeleton-based action recognition.
- **Fix cross-validation splits**: massive fold variance (RightTurn ranges from 43.2% in Fold 1 to 100% in Fold 4) strongly suggests data leakage or non-participant-level splits.
- **Use full skeleton for maneuver prediction**: the standalone maneuver predictor uses only 2D trajectory (input_size=2), discarding all 33-joint skeleton data.
- **Add data augmentation**: no augmentation is currently applied. Joint noise injection, temporal cropping, and speed perturbation are standard for skeleton data.
- **Fix training instability**: no LR scheduling (commented out), no gradient clipping, no class weighting for imbalanced data.
- **Fix code bugs**: `"\n".join(string)` bug, `epoch_val_loss` used before definition, hardcoded `cuda:0`/`cuda:1`, only fold 5 executed in training loops.

---

## 1. Current System Overview

The project implements a **two-stage hierarchical pipeline** for predicting cyclist maneuvers from skeleton data:

```
Stage 1: Skeleton Extraction (MediaPipe, 33 joints x 2 coords = 66 features)
    |
    v
Stage 2: Gesture Classification (30-frame sliding window)
    |-- Explicit gestures: Left_Hand, Right_Hand, Neutral (arm signals)
    |-- Implicit gestures: Left_Look, Left_Overshoulder, Right_Look, Neutral (head/body cues)
    |
    v
Stage 3: Maneuver Prediction (720 gesture predictions -> LSTM -> maneuver class)
    |-- Classes: Crossing, LeftTurn, RightTurn
```

**Architecture**: All models are Bi-LSTM with dropout and a final FC layer. The combined model runs a sliding window of 30 frames over 749-frame sequences, producing 720 gesture predictions that are mapped to scalar weights and fed to another LSTM.

**Model variants**:
- `CombinedModel` (weighting_set): maps gesture classes to fixed scalar weights (-2, -1, 0, +1), LSTM input_size=2
- `CombinedModel_HiddenWeighting` (weighting_NN): passes Bi-LSTM hidden states (256-dim) directly, LSTM input_size=512
- `CombinedModel_Explicit`: ablation using only explicit gestures, input_size=1

---

## 2. Current Performance

### 2.1 Explicit Gesture Classification (Fold 5 test set)

| Ground Truth | Left_Hand | Right_Hand | Neutral |
|---|---|---|---|
| **Left_Hand** | **100.0%** | 0.0% | 0.0% |
| **Right_Hand** | 0.0% | **100.0%** | 0.0% |
| **Neutral** | 2.4% | 1.1% | **96.5%** |

Verdict: **Strong**. Explicit arm signals are visually distinct and well-classified.

### 2.2 Implicit Gesture Classification (Fold 5 test set)

| Ground Truth | Left_Overshoulder | Left_Look | Right_Look | Neutral |
|---|---|---|---|---|
| **Left_Overshoulder** | **73.3%** | 2.8% | 12.6% | 11.3% |
| **Left_Look** | 8.5% | **25.9%** | 0.0% | **65.6%** |
| **Right_Look** | 0.0% | 0.0% | **98.0%** | 2.0% |
| **Neutral** | 3.3% | 6.2% | 15.7% | **74.8%** |

Verdict: **Near-broken**. Left_Look is classified as Neutral 65.6% of the time. Only Right_Look (98.0%) works reliably. The model has essentially learned to default to Neutral for ambiguous head movements.

### 2.3 Standalone Maneuver Prediction (Bi-LSTM on 2D trajectory, per-fold)

| Fold | Crossing | LeftTurn | RightTurn |
|---|---|---|---|
| **Fold 1** | 99.7% | 93.4% | **43.2%** |
| **Fold 2** | 100.0% | 85.3% | 100.0% |
| **Fold 3** | 99.5% | 93.4% | 100.0% |
| **Fold 4** | 100.0% | 93.4% | 100.0% |
| **Fold 5** | **32.2%** | 89.4% | 99.6% |

Verdict: **Highly unstable across folds**. RightTurn swings from 43.2% to 100%, Crossing from 32.2% to 100%. This variance is far too large for a 5-fold CV and indicates either data leakage in some folds, non-participant-level splits, or extreme class imbalance per fold.

### 2.4 Combined Model Results (from publication figures, fold 5)

| Model Variant | Crossing | LeftTurn | RightTurn |
|---|---|---|---|
| **Proposed (weighting_set + Ex_Im)** | 90.2% | 93.3% | 100.0% |
| **Previous research baseline** | 82.8% | 85.3% | 84.0% |
| **Ablation 1 (Explicit only)** | 90.2% | 93.1% | 57.5% |
| **Ablation 2 (Weighting NN)** | 78.2% | 73.6% | 99.8% |

Verdict: The proposed model shows improvement over the baseline, but Ablation 1 vs. Proposed shows that implicit gestures primarily help RightTurn (57.5% -> 100%). Given that the implicit classifier is near-broken, this improvement may be fragile.

---

## 3. Critical Issues (Ranked by Severity)

### 3.1 Implicit Gesture Classifier is Non-Functional

**Severity: Critical** | **Files**: `newsplit_immodel_definition.py`, `newsplit_model_training.py`

Left_Look at 25.9% accuracy is worse than a uniform-random baseline (25% for 4 classes). The 65.6% misclassification as Neutral means the model has collapsed to a majority-class bias. Root causes:

- **No class weighting**: `CrossEntropyLoss()` is used without weights despite severe class imbalance
- **Subtle head movements**: Left_Look and Neutral differ by small head rotations that 2D MediaPipe keypoints struggle to capture
- **Identical architecture to explicit**: the same Bi-LSTM architecture is used for both explicit (arm signals) and implicit (head orientation) gestures, but these are fundamentally different motion patterns
- **No spatial modeling**: head orientation requires understanding relative positions between nose, ears, and shoulders, which LSTM cannot model spatially

### 3.2 Cross-Validation Splits Are Likely Flawed

**Severity: Critical** | **Files**: `ManueverPrediction/model_training.py`, all dataloader files

The fold variance (RightTurn: 43.2%-100%, Crossing: 32.2%-100%) is abnormally high. Possible causes:

- **Non-participant-level splits**: if the same cyclist appears in both train and test, the model memorizes individual riding patterns. The data loading code shows no evidence of participant-level grouping.
- **Class imbalance per fold**: no stratification is visible in the fold creation. Some folds may have very few RightTurn or Crossing samples.
- **LabelEncoder fit independently per split**: in `ManueverPrediction_Combined/dataloader.py:19`, `LabelEncoder.fit_transform()` is called separately for train, val, and test. If a class is missing from one split, the label mapping will be inconsistent (the assertion on line 48 should catch this, but it's fragile).

### 3.3 Vanilla Bi-LSTM Cannot Model Skeleton Topology

**Severity: High** | **Files**: all `model_definition.py` files

The Bi-LSTM treats the 66-dimensional input as a flat vector, ignoring:
- **Joint connectivity**: the shoulder-elbow-wrist chain has spatial meaning that a flat vector destroys
- **Symmetry**: left/right body sides have mirrored structure
- **Hierarchical structure**: torso joints are more stable than extremity joints

Graph-based models (ST-GCN, CTR-GCN) explicitly encode the skeleton adjacency matrix, achieving 10-20% higher accuracy on standard benchmarks.

### 3.4 Maneuver Prediction Discards Skeleton Data

**Severity: High** | **Files**: `ManueverPrediction/model_training.py:180`

The standalone maneuver predictor uses `input_size=2` (the 2D gesture weight trajectory), throwing away the full 66-dimensional skeleton data. The combined model compounds this by reducing 66 dimensions to a single scalar weight per gesture type.

### 3.5 No Data Augmentation

**Severity: Medium** | All training files

No augmentation is applied anywhere in the pipeline. For skeleton data, standard augmentations include:
- Joint position noise (Gaussian, sigma=0.01-0.05)
- Temporal cropping and resampling
- Random rotation around vertical axis
- Speed perturbation (temporal stretching)
- Joint dropout (randomly zero out joints)

### 3.6 Training Configuration Issues

**Severity: Medium** | `newsplit_model_training.py`, `ManueverPrediction_Combined/model_training.py`

- **No LR scheduling**: all training uses fixed learning rate. Cosine annealing or step decay would improve convergence.
- **No gradient clipping**: LSTMs are prone to gradient explosion. `torch.nn.utils.clip_grad_norm_()` is not used anywhere.
- **Early stopping commented out or inconsistent**: gesture training uses patience=40 (very high for 80 epochs), combined model training has no early stopping in `train_fold()`.
- **Model selection criterion mismatch**: gesture training selects by best val loss (line 154), combined training by best F1 (line 147), standalone maneuver training by best val loss (line 145). The comments say "best F1" but code says `epoch_val_loss < best_val_loss`.

### 3.7 CombinedModel Sequential Inference Bottleneck

**Severity: Medium** | `ManueverPrediction_Combined/model_definition.py:246`

The CombinedModel forward pass runs a Python `for` loop over 720 sliding windows, executing two separate model forward passes per window. For a batch of 16, this means **720 x 2 = 1,440 sequential CUDA kernel launches** per sample. This could be vectorized by:
1. Extracting all windows as a batched tensor using `unfold()`
2. Running a single batched forward pass per gesture model

---

## 4. Code Quality Issues

### 4.1 Bugs

| Bug | File | Line | Description |
|---|---|---|---|
| **`"\n".join(string)` splits characters** | `GestureClassification/main.py` | 31, 42 | `"\n".join("some string")` produces `"s\no\nm\ne\n ..."`. Should be `f.write(string)` directly. Same bug in `ManueverPrediction_Combined/model_training.py:383` and `ManueverPrediction/model_training.py:277`. |
| **`epoch_val_loss` used before definition** | `ManueverPrediction_Combined/model_training.py` | 50 | `print(f"... Val Loss: {epoch_val_loss:.4f}")` appears before `epoch_val_loss` is computed (line 63). Will crash on first epoch. |
| **Module-level GPU device** | `newsplit_exmodel_definition.py`, `newsplit_immodel_definition.py`, `ManueverPrediction/model_definition.py` | 6-9 | `device = torch.device("cuda:0")` is set at module import time and used in `forward()` for hidden state initialization. This breaks if the model is moved to a different device. Should use `x.device` instead. |
| **Hardcoded `cuda:1`** | `ManueverPrediction_Combined/model_training.py` | 177 | `torch.device("cuda:1")` assumes a multi-GPU system. Will silently use wrong GPU or fail. |
| **Only fold 5 is trained** | `GestureClassification/main.py` | 19 | `range(5, 6)` runs only fold 5. Same issue in `ManueverPrediction_Combined/main.py:45`. All reported results are from a single fold, not proper cross-validation. |
| **LabelEncoder fit per split** | `ManueverPrediction_Combined/dataloader.py` | 19 | Each split gets its own `LabelEncoder.fit_transform()`. If class distribution differs, label indices may not match (e.g., if a class is missing from test). |

### 4.2 Code Smells

- **94 lines of commented-out code**: `ManueverPrediction_Combined/model_definition.py:3-94` contains an entire commented-out class
- **Duplicated model instantiation**: `ManueverPrediction_Combined/model_training.py` has the same 4-way `if/elif` block for model creation at lines 227-236 and again at lines 275-285
- **Mixed Chinese/English comments**: Chinese comments throughout (`# 预训练的显式手势分类模型`, `# 获取当前模型运行的设备`) reduce accessibility
- **No configuration file**: hyperparameters are scattered across `main.py` function signatures and hardcoded in training scripts (hidden_size=128, num_layers=4, lr=0.001, etc.)
- **Pickle files for data storage**: `.pkl` files are not human-readable, not version-controllable, and Python-version-dependent. Consider HDF5 or PyTorch `.pt` files.
- **Duplicate `import time`**: `newsplit_model_training.py:3` and `:6`, `ManueverPrediction_Combined/model_training.py:3` and `:8`
- **Debug prints left in**: `OCELoss` class has `print(output)`, `print(target)`, `print(target_i)` on every forward pass (lines 341-350 of `newsplit_model_training.py`)

---

## 5. Architecture Improvements

### 5.1 Replace Bi-LSTM with ST-GCN/CTR-GCN for Gesture Classification

**Why**: Spatial-Temporal Graph Convolutional Networks model the skeleton as a graph where joints are nodes and bones are edges. This captures spatial relationships (e.g., shoulder-elbow-wrist chain) that flat LSTM input destroys.

**Recommended models** (in order of complexity):
1. **ST-GCN** (Yan et al., AAAI 2018) - The baseline for skeleton action recognition. Simple, well-understood, many open-source implementations.
2. **CTR-GCN** (Chen et al., ICCV 2021) - Channel-wise topology refinement. Learns dynamic graph structure per channel.
3. **HD-GCN** (Lee et al., AAAI 2023) - Hierarchical decomposition. Models joints at multiple spatial scales.

**Directly relevant work**: [XingyuJinTI/Skeleton-based-cyclist-hand-signal-and-attention-recognition](https://github.com/XingyuJinTI/Skeleton-based-cyclist-hand-signal-and-attention-recognition) - An ST-GCN implementation specifically for cyclist gesture recognition on skeleton data.

**Expected impact**: 10-20% improvement on implicit gestures, where spatial relationships between head/shoulder joints are critical.

### 5.2 Consider PoseC3D for Noise-Robust Recognition

**Why**: MediaPipe keypoints in outdoor cycling scenarios are noisy (occlusion, motion blur, varying angles). PoseC3D (Duan et al., CVPR 2022) converts skeleton keypoints to 3D heatmap volumes and applies 3D CNNs, making it inherently robust to keypoint noise.

- Achieves 92.8% on NTU RGB+D 120 (skeleton-only)
- Does not require a pre-defined graph structure
- Better handles missing/noisy joints than GCN approaches

### 5.3 Add Temporal Attention for Maneuver Prediction

**Why**: Not all frames in a 749-frame sequence are equally informative. A head turn at frame 200 is more predictive than a neutral pose at frame 500. Self-attention or temporal attention mechanisms allow the model to focus on discriminative segments.

**Options**:
- **Multi-head self-attention** on top of LSTM hidden states
- **SkateFormer** (Do et al., 2024) - partition-based attention for skeleton sequences
- **Simple temporal attention**: learnable query vector that attends over LSTM outputs

### 5.4 Use Full Skeleton Features for Maneuver Prediction

Instead of reducing 66 dimensions to 2 scalar gesture weights, feed the full skeleton sequence directly to the maneuver predictor. This preserves:
- Body lean angle (indicative of turning)
- Arm position over time (not just classified gesture type)
- Subtle postural cues that don't fall into discrete gesture categories

### 5.5 Vectorize Sliding Window Inference

Replace the sequential Python loop in `CombinedModel.forward()`:

```python
# Current: O(720) sequential forward passes
for i in range(0, skeleton_x.size(1) - 30 + 1):
    window_data = skeleton_x[:, i:i + 30, :]
    explicit_out = self.gesture_model_explicit(window_data)
    implicit_out = self.gesture_model_implicit(window_data)
    ...

# Proposed: single batched forward pass
windows = skeleton_x.unfold(1, 30, 1)  # [batch, 720, 66, 30] -> reshape
windows = windows.permute(0, 1, 3, 2).reshape(-1, 30, 66)  # [batch*720, 30, 66]
explicit_out = self.gesture_model_explicit(windows)  # single forward pass
implicit_out = self.gesture_model_implicit(windows)  # single forward pass
```

This would reduce 1,440 CUDA kernel launches to 2.

---

## 6. Training & Data Improvements

### 6.1 Class-Weighted Loss for Implicit Gestures

The implicit gesture dataset is clearly imbalanced (Left_Look is rarely predicted). Add class weights inversely proportional to class frequency:

```python
# Count samples per class from training set
class_counts = torch.bincount(train_labels)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()  # normalize
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

Alternatively, use **Focal Loss** (Lin et al., 2017) which down-weights easy examples and focuses on hard misclassifications.

### 6.2 Data Augmentation for Skeleton Sequences

```python
# Joint noise injection
skeleton += torch.randn_like(skeleton) * 0.02

# Temporal cropping (random start/end within sequence)
start = random.randint(0, 5)
skeleton = skeleton[start:start+30]

# Random rotation around vertical axis
theta = random.uniform(-15, 15) * pi / 180
rotation_matrix = torch.tensor([[cos(theta), -sin(theta)],
                                 [sin(theta), cos(theta)]])
# Apply to x,y pairs of each joint

# Speed perturbation (temporal interpolation)
speed_factor = random.uniform(0.8, 1.2)
skeleton = F.interpolate(skeleton.unsqueeze(0), scale_factor=speed_factor)

# Joint dropout
mask = torch.bernoulli(torch.ones(33) * 0.9)  # 10% dropout
skeleton = skeleton * mask.repeat_interleave(2)
```

### 6.3 Proper Participant-Level Stratified Cross-Validation

Ensure that all sequences from the same cyclist are in the same fold. This prevents the model from memorizing individual riding styles:

```python
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# participant_ids: array of participant identifiers for each sample
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in sgkf.split(X, y, groups=participant_ids):
    ...
```

### 6.4 Learning Rate Scheduling

```python
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Or simple step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Call after each epoch
scheduler.step()
```

### 6.5 Gradient Clipping

```python
# Add after loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 6.6 Multi-Scale Temporal Windows

Instead of a fixed 30-frame window, use multiple window sizes to capture gestures at different speeds:

```python
window_sizes = [15, 30, 60]  # short, medium, long gestures
features = []
for ws in window_sizes:
    windows = skeleton.unfold(1, ws, stride)
    feat = model(windows)
    features.append(feat)
combined = torch.cat(features, dim=-1)
```

---

## 7. SOTA Context & Relevant Work

### 7.1 Skeleton-Based Action Recognition

| Model | Year | Venue | NTU RGB+D 120 (X-Sub) | Key Innovation |
|---|---|---|---|---|
| ST-GCN | 2018 | AAAI | 70.7% | First GCN for skeleton |
| MS-G3D | 2020 | CVPR | 86.9% | Multi-scale graph convolution |
| CTR-GCN | 2021 | ICCV | 88.9% | Channel-wise topology refinement |
| PoseC3D | 2022 | CVPR | 92.8% | 3D heatmap volumes + 3D CNN |
| HD-GCN | 2023 | AAAI | 89.7% | Hierarchical joint decomposition |
| SkateFormer | 2024 | - | 90.1% | Partition-based skeleton transformer |

### 7.2 Cyclist-Specific Work

- **Fang & Lopez (2019)**: "Intention Recognition of Pedestrians and Cyclists by 2D Pose Estimation" - Uses OpenPose 2D skeleton for VRU intention prediction. Shows that 2D pose alone can predict turning intention.
- **XingyuJinTI**: [Skeleton-based cyclist hand signal and attention recognition](https://github.com/XingyuJinTI/Skeleton-based-cyclist-hand-signal-and-attention-recognition) - ST-GCN applied to cyclist gesture recognition. Directly comparable to this project.
- **SRFic (2025)**: Social RNN for cyclist trajectory prediction, incorporating interaction modeling.

### 7.3 VRU Intention Prediction

- **JAAD/PIE datasets**: Standard benchmarks for pedestrian/cyclist intention prediction from video
- **IA-GRU**: Interaction-aware GRU for VRU trajectory prediction
- **Rasouli et al.**: Pedestrian intention estimation surveys covering skeleton-based approaches

---

## 8. Recommended Action Plan

### Priority 1: Quick Wins (1-2 days each)

These require minimal code changes but can significantly improve results:

1. **Fix the `"\n".join(string)` bug** in `GestureClassification/main.py:31,42` and `ManueverPrediction/model_training.py:277` and `ManueverPrediction_Combined/model_training.py:383`
2. **Fix `epoch_val_loss` reference before assignment** in `ManueverPrediction_Combined/model_training.py:50` (move the print after the validation block)
3. **Replace hardcoded `cuda:0`/`cuda:1`** with `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and use `x.device` in model forward methods
4. **Add class-weighted CrossEntropyLoss** for implicit gesture training
5. **Re-enable LR scheduling** (cosine annealing or step decay)
6. **Add gradient clipping** (`clip_grad_norm_`, max_norm=1.0)
7. **Run all 5 folds**: change `range(5, 6)` to `range(1, 6)` in `GestureClassification/main.py:19` and `ManueverPrediction_Combined/main.py:45`
8. **Create a config file** (YAML or dataclass) for all hyperparameters

### Priority 2: Medium Effort (1-2 weeks each)

These require moderate refactoring:

1. **Implement participant-level stratified CV splits** - requires knowing which sequences belong to which cyclist
2. **Replace Bi-LSTM with ST-GCN** for gesture classification - use the [XingyuJinTI implementation](https://github.com/XingyuJinTI/Skeleton-based-cyclist-hand-signal-and-attention-recognition) as a starting point
3. **Add data augmentation** (joint noise, temporal cropping, rotation)
4. **Vectorize sliding window inference** using `unfold()` for 100x+ speedup
5. **Feed full skeleton features** to maneuver prediction instead of 2D gesture weights
6. **Clean up codebase**: remove commented-out code, consolidate duplicated model instantiation, translate Chinese comments

### Priority 3: Larger Effort (1+ months)

These are significant architectural changes:

1. **Implement temporal attention/transformer** for maneuver prediction
2. **Multi-scale temporal windows** (15, 30, 60 frames)
3. **End-to-end training**: instead of frozen gesture models feeding into a maneuver LSTM, train the entire pipeline jointly with gradient flow through all stages
4. **Evaluate PoseC3D** as an alternative to GCN (more robust to MediaPipe noise)
5. **Benchmark on JAAD/PIE** for broader comparison with VRU intention prediction literature

---

## 9. References

1. Yan, S., Xiong, Y., & Lin, D. (2018). "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition." AAAI.
2. Liu, Z., et al. (2020). "Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition" (MS-G3D). CVPR.
3. Chen, Y., et al. (2021). "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition" (CTR-GCN). ICCV.
4. Duan, H., et al. (2022). "Revisiting Skeleton-based Action Recognition" (PoseC3D). CVPR.
5. Lee, J., et al. (2023). "Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition" (HD-GCN). AAAI.
6. Do, J., et al. (2024). "SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition."
7. Fang, Z., & Lopez, A.M. (2019). "Intention Recognition of Pedestrians and Cyclists by 2D Pose Estimation." IEEE ITSC.
8. Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
9. XingyuJinTI. "Skeleton-based cyclist hand signal and attention recognition." GitHub. https://github.com/XingyuJinTI/Skeleton-based-cyclist-hand-signal-and-attention-recognition
