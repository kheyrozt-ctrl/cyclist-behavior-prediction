# Changelog

## [Unreleased] - 2026-03-18

### Bug Fixes (Priority 1)
- Fix `"\n".join(string)` bug that split strings character-by-character in training time writers (5 locations across 4 files)
- Fix `epoch_val_loss` referenced before assignment in `ManueverPrediction_Combined/model_training.py`
- Fix hardcoded `cuda:1` device in `ManueverPrediction_Combined/model_training.py` and `model_evaluation.py`
- Fix module-level GPU device variable in 4 model definition files; now uses `x.device` for correct multi-device support
- Fix fold loop ranges from `range(5, 6)` to `range(1, 6)` to run all 5 folds (2 files)
- Fix hardcoded gesture model paths in `ManueverPrediction_Combined/main.py` to use `fold_number`

### Data Pipeline Fixes (Priority 2)
- Fix LabelEncoder fit-per-split bug in `ManueverPrediction_Combined/dataloader.py` — now fits encoder on training data only and uses `transform()` for val/test to ensure consistent label mapping
- Fix OrdinalEncoder usage in `GestureClassification/newsplit_dataloader.py` — fits on training data, reuses encoder for val/test

### Training Improvements
- Add class-weighted `CrossEntropyLoss` for gesture training to address class imbalance (Left_Look at 25.9%)
- Add cosine annealing LR scheduler (`CosineAnnealingLR`) to all training scripts
- Add gradient clipping (`max_norm=1.0`) to all training loops
- Add skeleton data augmentation (joint noise, joint dropout, random rotation) to gesture training

### Performance Improvements
- Vectorize sliding window inference in `ManueverPrediction_Combined/model_definition.py` using `torch.Tensor.unfold()` — replaces 720 sequential Python-loop forward passes with batched processing in configurable chunks
- Precompute gesture-to-weight lookup tables as registered buffers — replaces per-window Python string comparisons with vectorized tensor indexing
- Remove unnecessary implicit model forward pass in `CombinedModel_Explicit`

### Code Quality
- Create centralized `config.py` with dataclass-based hyperparameter configuration
- Extract `_create_combined_model()` helper to eliminate duplicated 4-way if/elif model instantiation blocks in `ManueverPrediction_Combined/model_training.py`
- Remove ~150 lines of commented-out dead code (old `CombinedModel` in model_definition.py, `train_fold_with_early_stopping` in training files)
- Remove debug `print()` statements (OCELoss debug prints, `print('changed')`)
- Remove duplicate imports (`import time` x2, `from sklearn.metrics import confusion_matrix` x2)
- Translate all Chinese comments to English across all active files
- Remove stale commented-out code blocks and unused `__main__` sections

### Architecture Notes (Not Yet Implemented)
The following recommendations from IMPROVEMENT_RECOMMENDATIONS.md require larger architectural changes:
- Replace Bi-LSTM with ST-GCN/CTR-GCN for gesture classification (models skeleton graph topology)
- Implement participant-level stratified cross-validation splits (requires participant ID metadata)
- Feed full 66-dim skeleton features to maneuver prediction instead of 2D gesture weights
- Add temporal attention/transformer for maneuver prediction
- Multi-scale temporal windows (15, 30, 60 frames)
- End-to-end training with gradient flow through all stages
