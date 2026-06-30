from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GestureClassification.dataloader import (
    class_names_for_branch,
)
from GestureClassification.model_evaluation import validate_model
from GestureClassification.newsplit_head_model_definition import HeadBiLSTMClassifier
from GestureClassification.newsplit_leg_model_definition import LegBiLSTMClassifier
from GestureClassification.newsplit_upper_model_definition import UpperBiLSTMClassifier


DEFAULT_RANDOM_STATE = 20260620


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 获取 DataLoader 的第一个 batch。
def get_input_output_size(dataloader):
    inputs, labels = next(iter(dataloader))
    all_labels = dataloader.dataset.tensors[1] if hasattr(dataloader.dataset, "tensors") else labels
    # 获取 input_size 和 output_size。
    return int(inputs.shape[-1]), int(torch.unique(all_labels).numel())


# Function to train a single fold.
def train_fold(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    patience=40,
):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_model_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    model.to(device)
    for epoch in range(1, int(num_epochs) + 1):
        # Training phase.
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass.
            loss = criterion(model(inputs), labels)
            # Backward pass and optimization.
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        train_losses.append(running_loss / max(1, len(train_loader)))

        # Validation phase.
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += float(criterion(outputs, labels).item())
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        epoch_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(epoch_val_loss)
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
        if macro_f1 > best_f1:
            # Save the best model based on validation score.
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_epoch = epoch
            best_acc = 100.0 * acc
            best_f1 = macro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            # Check if early stopping should be triggered.
            if patience_counter >= int(patience):
                break
    return train_losses, val_losses, best_model_state, best_model_epoch, best_acc, best_f1, best_val_loss


# 加载训练好的 Stage1 模型。
def load_stage1_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    branch = str(ckpt.get("branch", "")).lower()
    class_names = tuple(ckpt["class_names"])
    model_classes = {
        "head": HeadBiLSTMClassifier,
        "upper": UpperBiLSTMClassifier,
        "leg": LegBiLSTMClassifier,
    }
    if branch not in model_classes:
        raise ValueError(f"unknown Stage1 branch in checkpoint: {checkpoint_path}")
    model = model_classes[branch](
        input_size=66,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_classes=len(class_names),
        classifier_dropout=float(cfg["classifier_dropout"]),
        lstm_dropout=float(cfg["lstm_dropout"]),
        pooling=str(cfg["pooling"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names, ckpt["mean"].astype("float32"), ckpt["std"].astype("float32"), int(ckpt["window_rows"])


# 定义每个分支 checkpoint 的路径。
def stage1_checkpoint_paths(stage1_dir: Path, fold: int) -> dict[str, Path]:
    paths = {}
    for branch in ("head", "upper", "leg"):
        candidates = [
            stage1_dir / "stage1" / branch / f"fold_{fold}" / f"best_model_fold_{fold}.pkl",
            stage1_dir / branch / f"fold_{fold}" / f"best_model_fold_{fold}.pkl",
            stage1_dir / "stage1" / branch / f"fold_{fold}" / "best_model.pt",
            stage1_dir / branch / f"fold_{fold}" / "best_model.pt",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"missing Stage1 checkpoint for {branch}: {candidates[0]}")
        paths[branch] = path
    return paths


# Main function to train models for all folds.
def train_all_folds(
    dataloaders,
    model_save_dir,
    model_class,
    fold_number,
    num_epochs=None,
    hidden_size=None,
    learning_rate=None,
    num_layers=None,
    patience=None,
    eval_period=1,
    *,
    branch="head",
    classifier_dropout=None,
    lstm_dropout=None,
    pooling=None,
    weight_decay=None,
    label_smoothing=None,
    device=None,
):
    del eval_period
    branch = str(branch).lower()
    final_parameters = {
        "head": {
            "hidden_size": 64,
            "num_layers": 2,
            "pooling": "last_output",
            "classifier_dropout": 0.25,
            "lstm_dropout": 0.0,
            "lr": 4e-4,
            "weight_decay": 1e-6,
            "label_smoothing": 0.10,
            "epochs": 140,
            "patience": 50,
        },
        "upper": {
            "hidden_size": 96,
            "num_layers": 4,
            "pooling": "mean",
            "classifier_dropout": 0.55,
            "lstm_dropout": 0.15,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "label_smoothing": 0.0,
            "epochs": 240,
            "patience": 100,
        },
        "leg": {
            "hidden_size": 96,
            "num_layers": 2,
            "pooling": "last_output",
            "classifier_dropout": 0.25,
            "lstm_dropout": 0.0,
            "lr": 3e-4,
            "weight_decay": 1e-5,
            "label_smoothing": 0.0,
            "epochs": 160,
            "patience": 80,
        },
    }
    if branch not in final_parameters:
        raise ValueError(f"unknown Stage1 branch: {branch}")
    cfg = dict(final_parameters[branch])
    parameter_updates = {
        "epochs": num_epochs,
        "hidden_size": hidden_size,
        "lr": learning_rate,
        "num_layers": num_layers,
        "patience": patience,
        "classifier_dropout": classifier_dropout,
        "lstm_dropout": lstm_dropout,
        "pooling": pooling,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
    }
    cfg.update({key: value for key, value in parameter_updates.items() if value is not None})
    set_seed(DEFAULT_RANDOM_STATE)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取 input_size 和 output_size。
    input_size = int(next(iter(dataloaders["train"]))[0].shape[-1])
    class_names = tuple(dataloaders.get("class_names") or class_names_for_branch(branch))
    output_size = len(class_names)
    # Define the model.
    model = model_class(
        input_size=input_size,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_classes=output_size,
        classifier_dropout=float(cfg["classifier_dropout"]),
        lstm_dropout=float(cfg["lstm_dropout"]),
        pooling=str(cfg["pooling"]),
    ).to(device)
    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    result = train_fold(
        model,
        dataloaders["train"],
        dataloaders["val"],
        criterion,
        optimizer,
        int(cfg["epochs"]),
        device,
        patience=int(cfg["patience"]),
    )
    model.load_state_dict(result[2])
    # Save the best model.
    out = Path(model_save_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "branch": branch,
            "fold": int(fold_number),
            "class_names": list(class_names),
            "mean": np.asarray(dataloaders["mean"], dtype="float32"),
            "std": np.asarray(dataloaders["std"], dtype="float32"),
            "window_rows": int(next(iter(dataloaders["train"]))[0].shape[1]),
        },
        checkpoint_path,
    )
    counts = {}
    for subset in ("train", "val", "test"):
        labels = dataloaders[subset].dataset.tensors[1].cpu().numpy()
        counts[subset] = {name: int((labels == index).sum()) for index, name in enumerate(class_names)}
    # 使用最佳模型在验证集和测试集上进行评估。
    summary = {
        "branch": branch,
        "fold": int(fold_number),
        "counts": counts,
        "best_epoch": int(result[3]),
        "best_val_macro_f1": float(result[5]),
        "checkpoint": str(checkpoint_path),
        "config": cfg,
        "val": validate_model(model, dataloaders["val"], class_names, device),
        "test": validate_model(model, dataloaders["test"], class_names, device),
    }
    return model, summary
