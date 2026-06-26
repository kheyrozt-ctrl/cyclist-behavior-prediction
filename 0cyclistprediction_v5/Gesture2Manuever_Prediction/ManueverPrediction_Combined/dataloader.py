from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


MANEUVER_CLASSES = ("straight", "yield", "overtake")
MANEUVER_CATEGORIES = [list(MANEUVER_CLASSES)]


# Load data from a pickle file and create a DataLoader.
def create_dataloader(data_file, batch_size, shuffle=True):
    # 从 pickle 文件中加载数据。
    with Path(data_file).open("rb") as handle:
        data = pickle.load(handle)
    if not data:
        raise ValueError(f"no samples in {data_file}")

    # 将数据分为特征（X）和标签（y）。
    X = np.stack([np.asarray(item[0], dtype="float32") for item in data], axis=0)
    labels = [str(item[1]) for item in data]

    # 使用固定类别顺序对标签进行编码。
    class_to_idx = {label: index for index, label in enumerate(MANEUVER_CLASSES)}
    y = np.asarray([class_to_idx[label] for label in labels], dtype="int64")
    label_encoder = SimpleNamespace(classes_=np.asarray(MANEUVER_CLASSES, dtype=object))

    # 创建 TensorDataset 和 DataLoader。
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), label_encoder


# Create DataLoaders for one fold.
def create_and_save_dataloaders(data_dir, fold_number, batch_size=32):
    data_dir = Path(data_dir)
    print(f"Creating DataLoader for fold {fold_number}...")
    # 定义每个 fold 数据的路径。
    paths = {
        subset: data_dir / f"fold_{fold_number}_{subset}_data.pkl"
        for subset in ("train", "val", "test")
    }
    dataloaders = {}
    label_encoder = None
    for subset, path in paths.items():
        # 为训练、验证和测试创建 DataLoader。
        loader, current_encoder = create_dataloader(path, batch_size, shuffle=subset == "train")
        if label_encoder is None:
            label_encoder = current_encoder
        elif label_encoder.classes_.tolist() != current_encoder.classes_.tolist():
            raise ValueError("label encoders for train, val, and test do not match")
        dataloaders[subset] = loader

    dataloaders["label_encoder"] = label_encoder
    dataloaders["class_names"] = MANEUVER_CLASSES
    return dataloaders
