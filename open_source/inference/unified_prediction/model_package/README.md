##  environment: ## 

- Python: 3.10.19
- PyTorch: 2.8.0+cu129
- CUDA: 12.9


## Files: ## 

- `head_model.pt`: Stage1 head model / 第一阶段头部动作模型
- `upper_model.pt`: Stage1 upper-limb model / 第一阶段上肢动作模型
- `leg_model.pt`: Stage1 leg model / 第一阶段腿部动作模型
- `stage2_maneuver_model.pt`: Stage2 maneuver model / 第二阶段最终决策模型
- `model_config.json`: model metadata / 模型输入输出、类别顺序和特征维度说明


## Input notes: ## 

1 处理好的66维骨架 → 2 切出各分支 1 秒窗口 → 3 跑 Stage1 得到 (logits, feature) → 4 把所有窗口的 feature 按时间拼接成 120 个 640 维向量 → 5 输入 Stage2 → 6 `argmax(logits, dim=1)`，得到 maneuver 预测。
其中4为暂定，其他整体大致基于Zheng Tian和Ana的代码和模型。



- The `.pt` files are TorchScript models and can be loaded with `torch.jit.load()`. 
- Stage1 head input shape: `[batch, 12, 66]`       //头部模型输入，1秒窗口内12个骨架采样点，每个采样点66维。
- Stage1 upper-limb input shape: `[batch, 12, 66]`       //上肢模型输入，1秒窗口内12个骨架采样点，每个采样点66维。
- Stage1 leg input shape: `[batch, 20, 66]`       //腿部模型输入，1秒窗口内20个骨架采样点，每个采样点66维。
- 66D = 33 keypoints x 2D coordinates `(x, y)`.       //66维 = 33个关键点 x 二维坐标 `(x, y)`。
- Each Stage1 model returns `(logits, feature)`.       //`logits` 是分类得分，`feature` 是给第二阶段使用的隐藏特征。
- 模型本身不做严格的shape 检查，须保证输入 shape 正确。第一阶段三个模型分别输出 gesture classification logits，并同时输出 hidden feature，作为第二阶段 Stage2 的输入特征。

- Stage2 input shape: `[batch, 120, 640]`       //第二阶段模型暂定为接收三个第一阶段 hidden feature 拼接后的序列，第二阶段输入为10秒、12Hz的隐藏特征序列。
- Stage2 feature layout is `[head_feature_256 | upper_feature_192 | leg_feature_192]`，即 `640 = 256 + 192 + 192`。
- Stage2 输出为 straight / yield / overtake 的 logits，应用 softmax 后取最大概率类别。


##  Class order:  ## 

- head: `Left_Look`, `Right_Look`, `neutral_head`       //头部左看、头部右看、头部无动作。
- upper_limb: `Upper_Limb_Left_Rotation`, `Upper_Limb_Right_Rotation`, `neutral_upper_limb`       //上肢左偏转、上肢右偏转、上肢无动作。
- leg: `Pedaling`, `neutral_leg`       //腿部踩踏、腿部无动作。
- stage2: `straight`, `yield`, `overtake`       //最终决策：直行、让行、超车，`overtake` 已合并为左侧超车和右侧超车。


##  Output: ## 

- Use `argmax(logits, dim=1)` to obtain the predicted class index.       //使用 `argmax(logits, dim=1)` 得到预测类别编号。


## Real-time inference / 实时推理入口 ##

This package now includes `bus_stop_predict.py`, which connects the TorchScript models to the existing real-time pose pipeline.

```bash
cd ~/mathias_ws/model_package
python3 bus_stop_predict.py
```

Useful options:

```bash
# Use the fast trt_pose backend from ../pose_detection
python3 bus_stop_predict.py --pose trt

# Use a normal OpenCV webcam instead of RealSense
python3 bus_stop_predict.py --camera webcam --webcam-index 0

# Run over SSH/headless and save annotated output
python3 bus_stop_predict.py --headless -o bus_stop_output.avi

# Swap upper-limb left/right display labels if that branch appears reversed
python3 bus_stop_predict.py --swap-upper-labels

# Current TorchScript models run Stage1/Stage2 on CPU
python3 bus_stop_predict.py --device cpu
```

Runtime behavior:

- Pose input is still 66D skeleton data in MediaPipe 33-keypoint order.
- Head and upper-limb Stage1 windows are sampled at about 12 Hz for 12 samples.
- Leg Stage1 windows are sampled at about 20 Hz for 20 samples.
- Stage2 starts after 120 concatenated Stage1 feature vectors are collected, so the first final maneuver prediction appears after roughly 10 seconds of valid pose detections.
- Stage1 labels and Stage2 maneuver probabilities are drawn on the video frame.
- The current TorchScript package creates LSTM hidden states on CPU, so Stage1/Stage2 model inference is forced to CPU unless the models are re-exported with device-aware hidden states.
- If head direction is correct but upper-limb left/right appears reversed, use `--swap-upper-labels`. This only changes the displayed Stage1 upper-limb label and does not alter the Stage2 feature sequence.


## Evaluation status

This repository does not publish validated accuracy values for these packaged
legacy checkpoints. Report model quality only from a reproducible evaluation
that records the dataset revision, participant fold, code commit, configuration,
checkpoint hash, and per-class metrics.



