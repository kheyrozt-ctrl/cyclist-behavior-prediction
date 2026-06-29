from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

try:
    from .model_evaluation import validate_model
    from .newsplit_dataloader import class_names_for_branch
except ImportError:
    from model_evaluation import validate_model
    from newsplit_dataloader import class_names_for_branch


DEFAULT_RANDOM_STATE = 20260620


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts) -> int:
    value = 2166136261
    for byte in "|".join(str(part) for part in parts).encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value % 2_000_000_000)


def get_input_output_size(dataloader):
    inputs, labels = next(iter(dataloader))
    all_labels = dataloader.dataset.tensors[1] if hasattr(dataloader.dataset, "tensors") else labels
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
        running_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass.
            loss = criterion(model(inputs), labels)
            # Backward pass and optimization.
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(inputs.size(0))
            running_count += int(inputs.size(0))
        train_losses.append(running_loss / max(1, running_count))

        # Validation phase.
        model.eval()
        val_loss = 0.0
        val_count = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += float(criterion(outputs, labels).item()) * int(inputs.size(0))
                val_count += int(inputs.size(0))
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        epoch_val_loss = val_loss / max(1, val_count)
        val_losses.append(epoch_val_loss)
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
        if epoch_val_loss < best_val_loss:
            # Save the best model based on validation loss.
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
                print(f"Early Stopping triggered at epoch {epoch}")
                break
        print(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {acc:.4f}, Val F1: {macro_f1:.4f}"
        )
    return train_losses, val_losses, best_model_state, best_model_epoch, best_acc, best_f1, best_val_loss


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
    train_seed = stable_seed("final_bilstm_paired", branch, DEFAULT_RANDOM_STATE, fold_number)
    set_seed(train_seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get input_size and output_size from the DataLoader.
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
    batch_size = int(cfg.get("batch_size", 32))
    train_generator = torch.Generator().manual_seed(int(train_seed) + 1)
    train_loader = DataLoader(
        dataloaders["train"].dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        dataloaders["val"].dataset,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataloaders["test"].dataset,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    print(f"Training model for fold {fold_number} ({branch})...")
    result = train_fold(
        model,
        train_loader,
        val_loader,
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
    checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "branch": branch,
        "fold": int(fold_number),
        "class_names": list(class_names),
        "mean": np.asarray(dataloaders["mean"], dtype="float32"),
        "std": np.asarray(dataloaders["std"], dtype="float32"),
        "window_rows": int(next(iter(dataloaders["train"]))[0].shape[1]),
    }
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, out / f"best_model_fold_{fold_number}.pkl")
    # 保存 DataLoader 相关变量以供下个脚本使用。
    with (out / f"dataloader_fold_{fold_number}.pkl").open("wb") as handle:
        pickle.dump((train_loader, val_loader, test_loader, dataloaders.get("encoder")), handle)
    counts = {}
    for subset in ("train", "val", "test"):
        labels = dataloaders[subset].dataset.tensors[1].cpu().numpy()
        counts[subset] = {name: int((labels == index).sum()) for index, name in enumerate(class_names)}
    val_metrics = validate_model(model, val_loader, class_names, device)
    test_metrics = validate_model(model, test_loader, class_names, device)
    train_metrics = validate_model(model, train_loader, class_names, device)
    summary = {
        "branch": branch,
        "fold": int(fold_number),
        "counts": counts,
        "best_epoch": int(result[3]),
        "best_val_loss": float(result[6]),
        "best_val_macro_f1": float(result[5]),
        "checkpoint_selection": "validation_loss_only",
        "checkpoint": str(checkpoint_path),
        "config": cfg,
        "val": val_metrics,
        "test": test_metrics,
    }
    # 使用最佳模型在测试集上进行评估。
    print(
        f"Best model for fold {fold_number} saved at epoch {int(result[3])} "
        f"with Val Loss: {float(result[6]):.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
        f"Val f1: {val_metrics['macro_f1']:.4f}."
    )
    print(f"Test Accuracy: {100 * test_metrics['accuracy']:.2f}%")
    print(f"Train Accuracy: {100 * train_metrics['accuracy']:.2f}%")
    print(f"Val Accuracy: {100 * val_metrics['accuracy']:.2f}%")
    return model, summary
