from __future__ import annotations

import torch
from torch import nn


# 定义 leg 分支 BiLSTM 模型。
class LegBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_classes: int = 2,
        classifier_dropout: float = 0.25,
        lstm_dropout: float = 0.0,
        pooling: str = "last_output",
        dropout: float | None = None,
    ):
        super().__init__()
        if dropout is not None:
            classifier_dropout = dropout
            lstm_dropout = dropout if num_layers > 1 else 0.0
        self.pooling = pooling
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 全连接层，将 LSTM 的输出映射到手势类别空间。
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Forward propagate LSTM.
        out, (h, _) = self.lstm(x)
        # Decode the pooled hidden representation.
        if self.pooling == "mean":
            return out.mean(dim=1)
        if self.pooling == "final_hidden":
            return torch.cat([h[-2], h[-1]], dim=1)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        feat = self.encode(x)
        if return_hidden:
            return feat
        return self.fc(self.dropout(feat))
