from __future__ import annotations

import numpy as np
import torch
from torch import nn


TARGET_FRAMES = 120
BRANCHES = ("head", "upper", "leg")
BRANCH_SOURCE_FRAMES = {"head": 12, "upper": 12, "leg": 12}
BRANCH_OUTPUT_FRAMES = {"head": 12, "upper": 12, "leg": 20}
FEATURE_TIMEPOINTS = TARGET_FRAMES - 12 + 1
FEATURE_DIM = 512


# 定义 Stage2 投影 BiLSTM 模型。
class ProjectedStage2BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        projection_dim: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.65,
        pooling: str = "last_output",
    ):
        super().__init__()
        self.pooling = "last_output" if pooling == "last" else pooling
        # 将 Stage1 hidden feature 投影到低维表示。
        self.projection = nn.Sequential(
            nn.Linear(input_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(min(max(dropout * 0.5, 0.0), 0.5)),
        )
        # LSTM 层，输入是 Stage1 hidden feature 序列。
        self.lstm = nn.LSTM(
            projection_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # 全连接层，将 LSTM 的输出映射到 maneuver 预测类别空间。
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        z = self.projection(x)
        # LSTM 层前向传播。
        out, (h, _) = self.lstm(z)
        # 使用池化后的隐藏状态作为分类输入。
        if self.pooling == "mean":
            if lengths is None:
                feat = out.mean(dim=1)
            else:
                mask = torch.arange(out.shape[1], device=out.device)[None, :] < lengths[:, None]
                feat = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        elif self.pooling == "final_hidden":
            feat = torch.cat([h[-2], h[-1]], dim=1)
        else:
            if lengths is None:
                feat = out[:, -1, :]
            else:
                idx = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
                feat = out.gather(1, idx).squeeze(1)
        return self.fc(self.dropout(feat))


# 沿时间轴插值到分支模型需要的输入长度。
def _interpolate_time_axis_torch(array: torch.Tensor, target_len: int) -> torch.Tensor:
    source_len = int(array.shape[1])
    if source_len == int(target_len):
        return array
    target = torch.linspace(0, source_len - 1, int(target_len), device=array.device, dtype=array.dtype)
    left = target.floor().long()
    right = torch.clamp(left + 1, max=source_len - 1)
    alpha = (target - left.to(array.dtype)).view(1, int(target_len), 1)
    return (1.0 - alpha) * array[:, left, :] + alpha * array[:, right, :]


# 定义联合模型。
class CombinedModel(nn.Module):
    def __init__(
        self,
        gesture_models,
        feature_mean,
        feature_std,
        hidden_size=64,
        output_size=3,
        num_layers=1,
        dropout=0.65,
        *,
        projection_dim=16,
        pooling="last_output",
    ):
        super().__init__()
        # 预训练的 Stage1 分支手势分类模型。
        self.gesture_models = nn.ModuleDict({branch: value[0] for branch, value in gesture_models.items()})
        for branch, (_model, _classes, mean, std, model_rows) in gesture_models.items():
            expected_rows = BRANCH_OUTPUT_FRAMES[branch]
            if int(model_rows) != int(expected_rows):
                raise ValueError(f"{branch} checkpoint expects {model_rows} rows, got {expected_rows}")
            self.register_buffer(f"{branch}_mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 1, -1))
            std = np.where(np.abs(np.asarray(std, dtype="float32")) < 1e-6, 1.0, std).astype("float32")
            self.register_buffer(f"{branch}_std", torch.as_tensor(std, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32).view(1, 1, -1))
        self.stage2 = ProjectedStage2BiLSTM(
            input_size=FEATURE_DIM,
            projection_dim=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=output_size,
            dropout=dropout,
            pooling=pooling,
        )
        for module in self.gesture_models.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.gesture_models.eval()
        return self

    def set_feature_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        # 更新 Stage2 特征标准化参数。
        mean_tensor = torch.as_tensor(mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device).view(1, 1, -1)
        std = np.where(np.abs(np.asarray(std, dtype="float32")) < 1e-6, 1.0, std).astype("float32")
        std_tensor = torch.as_tensor(std, dtype=self.feature_std.dtype, device=self.feature_std.device).view(1, 1, -1)
        self.feature_mean.copy_(mean_tensor)
        self.feature_std.copy_(std_tensor)

    def _branch_input(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        # 滑动窗口操作。
        source_len = int(BRANCH_SOURCE_FRAMES[branch])
        target_len = int(BRANCH_OUTPUT_FRAMES[branch])
        max_start = int(TARGET_FRAMES) - source_len
        if x.shape[1] != int(TARGET_FRAMES):
            raise ValueError(f"expected {TARGET_FRAMES} frames, got {x.shape[1]}")
        windows = torch.stack([x[:, start : start + source_len, :] for start in range(max_start + 1)], dim=1)
        flat = windows.reshape(-1, source_len, x.shape[-1])
        flat = _interpolate_time_axis_torch(flat, target_len)
        mean = getattr(self, f"{branch}_mean")
        std = getattr(self, f"{branch}_std")
        return (flat - mean) / std

    @torch.no_grad()
    def extract_hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        # 初始化 Stage1 特征序列。
        parts = []
        for branch in BRANCHES:
            branch_input = self._branch_input(x, branch)
            # 使用 Stage1 分支模型生成 hidden feature。
            hidden = self.gesture_models[branch](branch_input, return_hidden=True)
            if isinstance(hidden, tuple):
                hidden = hidden[-1]
            parts.append(hidden.reshape(x.shape[0], FEATURE_TIMEPOINTS, -1))
        # 将 head/upper/leg 特征拼接。
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # 获取当前模型运行的设备并生成 Stage1 特征。
        with torch.no_grad():
            features = self.extract_hidden_features(x)
            features = (features - self.feature_mean) / self.feature_std
        return self.stage2(features, lengths)
