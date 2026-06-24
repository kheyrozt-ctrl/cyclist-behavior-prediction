from __future__ import annotations

import copy
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GestureClassification.model_training import load_stage1_model, stage1_checkpoint_paths
from ManueverPrediction_Combined.dataloader import MANEUVER_CLASSES
from ManueverPrediction_Combined.model_definition import CombinedModel, FEATURE_DIM


DEFAULT_RANDOM_STATE = 20260620


# Function to train a single fold with early stopping
def train_fold_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    patience,
    eval_period,
    device,
):
    train_losses, val_losses = [], []
    best_model_state = copy.deepcopy(model.state_dict())
    best_model_epoch = 0
    best_macro_f1 = -1.0
    stale = 0
    model.to(device)
    for epoch in range(1, int(num_epochs) + 1):
        # Training phase
        model.train()
        losses = []
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_losses.append(float(np.mean(losses)) if losses else 0.0)
        if epoch % max(1, int(eval_period)) != 0:
            continue
        # Validation phase
        metrics = validate_model(model, val_loader, MANEUVER_CLASSES, device)
        val_losses.append(float(metrics["loss"]))
        # Early stopping check
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_epoch = epoch
            stale = 0
        else:
            stale += 1
            # Check if early stopping should be triggered
            if stale >= int(patience):
                print(f"Early Stopping triggered at epoch {epoch}")
                break
        print(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
            f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}, "
            f"Val F1: {metrics['macro_f1']:.4f}"
        )
    return train_losses, val_losses, best_model_state, best_model_epoch, best_macro_f1


# Get the first batch from a DataLoader.
def get_input_output_size(dataloader):
    batch = next(iter(dataloader))
    inputs = batch[0]
    return int(inputs.shape[-1]), len(MANEUVER_CLASSES)


def validate_model(model, loader, class_names, device):
    model.eval()
    y_true, y_pred, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    # 运行验证。
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses.append(float(criterion(outputs, labels).item()))
            predicted = torch.argmax(outputs, dim=1)
            y_pred.extend(predicted.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
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
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            class_names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(class_names))
        },
    }


# Main function to train models for all folds
def train_all_folds(
    dataloaders=None,
    model_save_dir=None,
    fold_number=None,
    num_epochs=None,
    hidden_size=None,
    learning_rate=None,
    num_layers=None,
    patience=None,
    eval_period=1,
    *,
    parameter_updates=None,
    stage1_checkpoint_dir=None,
    device=None,
):
    del num_epochs, hidden_size, learning_rate, num_layers, patience, eval_period
    if dataloaders is None or model_save_dir is None or fold_number is None:
        raise ValueError("dataloaders, model_save_dir, and fold_number are required")
    if stage1_checkpoint_dir is None:
        raise ValueError("stage1_checkpoint_dir is required for online Stage2 training")
    return _train_all_folds_projected(
        dataloaders,
        model_save_dir,
        fold_number,
        parameter_updates=parameter_updates,
        stage1_checkpoint_dir=stage1_checkpoint_dir,
        device=device,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gesture_models(stage1_dir: Path, fold: int, device) -> dict:
    models = {}
    for branch, path in stage1_checkpoint_paths(stage1_dir, fold).items():
        model, classes, mean, std, model_rows = load_stage1_model(path, str(device))
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[branch] = (model, classes, mean, std, model_rows)
    return models


def compute_feature_normalizer(model, train_loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sums = None
    sums_sq = None
    count = 0
    # 统计训练集 Stage1 hidden feature 的均值和方差。
    with torch.no_grad():
        for inputs, _labels in train_loader:
            features = model.extract_hidden_features(inputs.to(device, dtype=torch.float32))
            flat = features.detach().cpu().numpy().astype("float64").reshape(-1, features.shape[-1])
            sums = flat.sum(axis=0) if sums is None else sums + flat.sum(axis=0)
            sums_sq = (flat * flat).sum(axis=0) if sums_sq is None else sums_sq + (flat * flat).sum(axis=0)
            count += int(flat.shape[0])
    mean = (sums / max(1, count)).astype("float32")
    variance = (sums_sq / max(1, count)) - (mean.astype("float64") ** 2)
    std = np.sqrt(np.maximum(variance, 0.0)).astype("float32")
    std = np.where(np.abs(std) < 1e-6, 1.0, std).astype("float32")
    return mean, std


def _train_all_folds_projected(
    dataloaders,
    model_save_dir,
    fold_number,
    parameter_updates=None,
    stage1_checkpoint_dir=None,
    device=None,
):
    train_seed = DEFAULT_RANDOM_STATE * 100 + int(fold_number)
    set_seed(train_seed)
    cfg = {
        "input_size": 512,
        "projection_dim": 16,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.65,
        "pooling": "last_output",
        "lr": 3e-4,
        "weight_decay": 3e-4,
        "epochs": 80,
        "patience": 16,
        "class_step_frames": {"straight": 2, "yield": 1, "overtake": 1},
        "dense_window_sec": 10.0,
        "stage2_window_frames": 120,
        "stage2_feature_timepoints": 109,
        "stage2_feature_policy": "branch_last_time_output_learned_projection_16d",
        "stage2_feature_mode": "online_stage1_hidden",
        "feature_normalization": "standard",
    }
    if parameter_updates:
        cfg.update({key: value for key, value in parameter_updates.items() if value is not None})
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取 input_size 和 output_size。
    _raw_input_size, output_size = get_input_output_size(dataloaders["train"])
    cfg["input_size"] = int(FEATURE_DIM)
    batch_size = int(getattr(dataloaders["train"], "batch_size", 32) or 32)
    generator = torch.Generator()
    generator.manual_seed(DEFAULT_RANDOM_STATE * 1000 + int(fold_number))
    active_loaders = {
        "train": DataLoader(
            dataloaders["train"].dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "val": DataLoader(dataloaders["val"].dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(dataloaders["test"].dataset, batch_size=batch_size, shuffle=False),
    }
    # Load pre-trained gesture models.
    gesture_models = load_gesture_models(Path(stage1_checkpoint_dir), int(fold_number), device)
    # Define the model.
    model = CombinedModel(
        gesture_models,
        np.zeros(FEATURE_DIM, dtype="float32"),
        np.ones(FEATURE_DIM, dtype="float32"),
        projection_dim=int(cfg["projection_dim"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        output_size=output_size,
        dropout=float(cfg["dropout"]),
        pooling=str(cfg["pooling"]),
    ).to(device)
    feature_mean, feature_std = compute_feature_normalizer(model, active_loaders["train"], device)
    model.set_feature_normalizer(feature_mean, feature_std)
    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.stage2.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    print(f"Training model for fold {fold_number}...")
    # Train the model.
    result = train_fold_with_early_stopping(
        model,
        active_loaders["train"],
        active_loaders["val"],
        criterion,
        optimizer,
        int(cfg["epochs"]),
        int(cfg["patience"]),
        1,
        device,
    )
    model.load_state_dict(result[2])

    # Save the best model.
    out = Path(model_save_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "best_model.pt"
    normalizer = {
        "mean": feature_mean,
        "std": feature_std,
        "feature_normalization": "standard",
    }
    np.save(out / "stage2_feature_mean.npy", np.asarray(normalizer["mean"], dtype="float32"))
    np.save(out / "stage2_feature_std.npy", np.asarray(normalizer["std"], dtype="float32"))
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "fold": int(fold_number),
            "class_names": list(MANEUVER_CLASSES),
            "feature_mean": np.asarray(normalizer.get("mean", []), dtype="float32"),
            "feature_std": np.asarray(normalizer.get("std", []), dtype="float32"),
            "stage1_checkpoint_dir": str(stage1_checkpoint_dir),
        },
        checkpoint_path,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "fold": int(fold_number),
            "class_names": list(MANEUVER_CLASSES),
            "feature_mean": np.asarray(normalizer.get("mean", []), dtype="float32"),
            "feature_std": np.asarray(normalizer.get("std", []), dtype="float32"),
            "stage1_checkpoint_dir": str(stage1_checkpoint_dir),
        },
        out / f"best_model_fold_{fold_number}.pkl",
    )
    # 保存 DataLoader 相关变量以供下个脚本使用。
    with (out / f"dataloader_fold_{fold_number}.pkl").open("wb") as handle:
        pickle.dump(
            (
                active_loaders["train"],
                active_loaders["val"],
                active_loaders["test"],
                device,
                dataloaders.get("label_encoder"),
            ),
            handle,
        )
    counts = {
        subset: {
            class_name: int((active_loaders[subset].dataset.tensors[1] == index).sum().item())
            for index, class_name in enumerate(MANEUVER_CLASSES)
        }
        for subset in ("train", "val", "test")
    }
    # 使用最佳模型在测试集上进行评估。
    test_metrics = validate_model(model, active_loaders["test"], MANEUVER_CLASSES, device)
    val_metrics = validate_model(model, active_loaders["val"], MANEUVER_CLASSES, device)
    train_metrics = validate_model(model, active_loaders["train"], MANEUVER_CLASSES, device)
    # 记录单个 input 经过模型的时间。
    inference_start = time.time()
    inference_count = 0
    model.eval()
    with torch.no_grad():
        for inputs, _labels in active_loaders["test"]:
            outputs = model(inputs.to(device))
            inference_count += int(outputs.shape[0])
    # 计算平均推理时间。
    average_inference_time = (time.time() - inference_start) / max(1, inference_count)
    summary = {
        "fold": int(fold_number),
        "counts": counts,
        "best_epoch": int(result[3]),
        "best_val_macro_f1": float(result[4]),
        "checkpoint": str(checkpoint_path),
        "config": cfg,
        "feature_normalizer": {
            "feature_normalization": str(normalizer.get("feature_normalization", "standard")),
            "mean_shape": list(np.asarray(normalizer.get("mean", [])).shape),
            "std_shape": list(np.asarray(normalizer.get("std", [])).shape),
        },
        "window_test": test_metrics,
        "val": val_metrics,
    }
    (out / f"inference_time_fold_{fold_number}.txt").write_text(
        f"Average inference time per input: {average_inference_time:.6f} seconds\n",
        encoding="utf-8",
    )
    print(
        f"Best model for fold {fold_number} saved at epoch {int(result[3])} "
        f"with Val Loss: {val_metrics['loss']:.4f}."
    )
    print(f"Evaluating on test set with best validation loss: {val_metrics['loss']:.4f}")
    print(f"Test Accuracy: {100 * test_metrics['accuracy']:.2f}%, Test F1: {test_metrics['macro_f1']:.2f}")
    print(f"Average inference time per input: {average_inference_time:.6f} seconds")
    print(f"Train Accuracy: {100 * train_metrics['accuracy']:.2f}%, Train F1: {train_metrics['macro_f1']:.2f}")
    print(f"Val Accuracy: {100 * val_metrics['accuracy']:.2f}%, Val F1: {val_metrics['macro_f1']:.2f}")
    return model, summary
