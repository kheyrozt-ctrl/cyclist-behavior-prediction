from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


# 计算分类指标。
def _classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) if y_true else 0.0
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
    }


# 定义验证函数。
def validate_model(model: nn.Module, loader: DataLoader, class_names: Sequence[str], device: str) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    # 运行验证。
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                logits = model(x.to(device))
            else:
                x, lengths, y, _meta = batch
                logits = model(x.to(device), lengths.to(device))
            y = y.to(device)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    out = _classification_metrics(y_true, y_pred, class_names)
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out
