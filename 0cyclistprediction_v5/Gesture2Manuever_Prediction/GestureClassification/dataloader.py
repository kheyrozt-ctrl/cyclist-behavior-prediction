from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, TensorDataset


HEAD_CLASSES = ("Left_Look", "Right_Look", "neutral")
UPPER_CLASSES = ("Upper_Limb_Left_Rotation", "Upper_Limb_Right_Rotation", "neutral")
LEG_CLASSES = ("Pedaling", "neutral")

# 自定义类别顺序用于 OrdinalEncoder。
LABEL_CATEGORIES = {
    "head": [list(HEAD_CLASSES)],
    "upper": [list(UPPER_CLASSES)],
    "leg": [list(LEG_CLASSES)],
}


def class_names_for_branch(branch: str) -> tuple[str, ...]:
    key = str(branch).lower()
    if key == "head":
        return HEAD_CLASSES
    if key == "upper":
        return UPPER_CLASSES
    if key == "leg":
        return LEG_CLASSES
    raise ValueError(f"unknown Stage1 branch: {branch}")


# 从 pickle 文件中加载数据。
def _read_pickle(data_file):
    with Path(data_file).open("rb") as handle:
        data = pickle.load(handle)
    if not data:
        raise ValueError(f"no samples in {data_file}")
    # 将数据分为特征（X）和标签（y）。
    arrays = [item[0].to_numpy() if isinstance(item[0], pd.DataFrame) else np.asarray(item[0]) for item in data]
    labels = [item[1] for item in data]
    return np.stack(arrays).astype("float32"), labels


# 函数：从 pickle 文件加载数据并创建 DataLoader。
def create_dataloader(data_file, batch_size, dataset_type, shuffle=True, mean=None, std=None):
    arrays, raw_labels = _read_pickle(data_file)
    key = str(dataset_type).strip().lower()
    if key not in LABEL_CATEGORIES:
        raise ValueError(f"unknown dataset_type: {dataset_type}")
    if mean is not None and std is not None:
        arrays = (arrays - np.asarray(mean, dtype="float32")[None, None, :]) / np.asarray(std, dtype="float32")[None, None, :]
    labels = np.asarray(raw_labels, dtype=object).reshape(-1, 1)
    # 使用 OrdinalEncoder 对标签进行编码。
    encoder = OrdinalEncoder(categories=LABEL_CATEGORIES[key])
    encoded = encoder.fit_transform(labels).astype("int64").reshape(-1)
    # 创建 TensorDataset。
    dataset = TensorDataset(
        torch.as_tensor(arrays, dtype=torch.float32),
        torch.as_tensor(encoded, dtype=torch.long),
    )
    # 创建 DataLoader。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), encoder


# 主函数：为每个 fold 创建 DataLoader。
def create_and_save_dataloaders(data_dir, dataset_type, fold_number, batch_size=32):
    data_dir = Path(data_dir)
    branch = str(dataset_type).lower()
    paths = {}
    for subset in ("train", "val", "test"):
        # 定义每个 fold 数据的路径。
        candidates = [
            data_dir / f"fold_{fold_number}_{branch}_{subset}_data.pkl",
            data_dir / f"fold_{fold_number}_{subset}_data.pkl",
        ]
        paths[subset] = next((candidate for candidate in candidates if candidate.exists()), None)
        if paths[subset] is None:
            raise FileNotFoundError(candidates[0])
    train_arrays, _ = _read_pickle(paths["train"])
    flat = train_arrays.reshape(-1, train_arrays.shape[-1])
    mean = flat.mean(axis=0).astype("float32")
    std = flat.std(axis=0).astype("float32")
    std[std < 1e-6] = 1.0
    loaders = {}
    encoder = None
    for subset in ("train", "val", "test"):
        # 为训练、验证和测试创建 DataLoader。
        loader, current_encoder = create_dataloader(
            paths[subset],
            batch_size,
            branch,
            shuffle=subset == "train",
            mean=mean,
            std=std,
        )
        loaders[subset] = loader
        encoder = encoder or current_encoder
    loaders["encoder"] = encoder
    loaders["mean"] = mean
    loaders["std"] = std
    loaders["class_names"] = class_names_for_branch(branch)
    return loaders
