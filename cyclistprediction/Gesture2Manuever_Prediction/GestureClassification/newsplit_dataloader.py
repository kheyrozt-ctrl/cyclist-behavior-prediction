import os
import sys
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import random

# Compatibility shim: pickle files created with numpy >= 2.0 reference numpy._core,
# which doesn't exist in numpy < 2.0. Map it to numpy.core so unpickling works.
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

# Fixed category orderings for OrdinalEncoder
explicit_categories = [['Right_Hand_Explicit', 'neutral_explicit', 'Left_Hand_Explicit']]
implicit_categories = [['Right_Look', 'neutral_implicit', 'Left_Overshoulder', 'Left_Look']]


class SkeletonAugmentation:
    """Data augmentation transforms for skeleton sequences."""

    def __init__(self, joint_noise_std=0.02, temporal_crop_ratio=0.9,
                 rotation_range=15.0, speed_range=(0.8, 1.2), joint_dropout_prob=0.1):
        self.joint_noise_std = joint_noise_std
        self.temporal_crop_ratio = temporal_crop_ratio
        self.rotation_range = rotation_range
        self.speed_range = speed_range
        self.joint_dropout_prob = joint_dropout_prob

    def __call__(self, skeleton):
        """Apply random augmentations to a skeleton tensor [seq_len, num_features]."""
        if random.random() < 0.5:
            skeleton = self.add_joint_noise(skeleton)
        if random.random() < 0.3:
            skeleton = self.apply_joint_dropout(skeleton)
        if random.random() < 0.3:
            skeleton = self.apply_rotation(skeleton)
        return skeleton

    def add_joint_noise(self, skeleton):
        """Add Gaussian noise to joint positions."""
        noise = torch.randn_like(skeleton) * self.joint_noise_std
        return skeleton + noise

    def apply_joint_dropout(self, skeleton):
        """Randomly zero out entire joints (pairs of x,y coordinates)."""
        num_joints = skeleton.shape[1] // 2
        mask = torch.bernoulli(torch.ones(num_joints) * (1 - self.joint_dropout_prob))
        # Each joint has 2 coordinates (x, y)
        mask = mask.repeat_interleave(2)
        return skeleton * mask.unsqueeze(0)

    def apply_rotation(self, skeleton):
        """Apply small random rotation to x,y coordinate pairs."""
        angle = random.uniform(-self.rotation_range, self.rotation_range) * (3.14159 / 180)
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))

        rotated = skeleton.clone()
        num_joints = skeleton.shape[1] // 2
        for j in range(num_joints):
            x_idx, y_idx = j * 2, j * 2 + 1
            x, y = skeleton[:, x_idx], skeleton[:, y_idx]
            rotated[:, x_idx] = x * cos_a - y * sin_a
            rotated[:, y_idx] = x * sin_a + y * cos_a
        return rotated


def create_dataloader(data_file, batch_size, dataset_type, shuffle=True, encoder=None, augment=False):
    """Load data from pickle file and create a DataLoader.

    If encoder is provided, reuses it for consistent encoding.
    If None, creates and fits a new encoder (should only be done for training set).
    If augment=True, applies skeleton augmentations to training data.
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X = [torch.tensor(x[0].values, dtype=torch.float32) if isinstance(x[0], pd.DataFrame)
         else torch.tensor(x[0], dtype=torch.float32) for x in data]  # Features: 30x66 matrices
    y = [x[1] for x in data]  # Labels: gesture type

    # Apply augmentation to training data
    if augment:
        aug = SkeletonAugmentation()
        X = [aug(x) for x in X]

    # Create or reuse encoder
    if encoder is None:
        if dataset_type == 'explicit':
            encoder = OrdinalEncoder(categories=explicit_categories)
        elif dataset_type == 'implicit':
            encoder = OrdinalEncoder(categories=implicit_categories)
        else:
            raise ValueError("Unknown dataset_type: choose either 'explicit' or 'implicit'")

    y = np.array(y).reshape(-1, 1)
    y_encoded = encoder.fit_transform(y).astype(int).flatten()
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    X_tensor = torch.stack(X)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, encoder


def create_and_save_dataloaders(data_dir, dataset_type, fold_number, batch_size=16):
    """Create DataLoaders for a given fold.

    Fits encoder on training data, reuses for val/test.
    Applies data augmentation only to training data.
    """
    train_data_file = os.path.join(data_dir, f'fold_{fold_number}_train_data.pkl')
    val_data_file = os.path.join(data_dir, f'fold_{fold_number}_val_data.pkl')
    test_data_file = os.path.join(data_dir, f'fold_{fold_number}_test_data.pkl')

    print(f"Creating DataLoader for fold {fold_number}...")
    # Fit encoder on training data, apply augmentation to training only
    train_loader, encoder = create_dataloader(train_data_file, batch_size, dataset_type, augment=True)
    val_loader, _ = create_dataloader(val_data_file, batch_size, dataset_type, shuffle=False, encoder=encoder)
    test_loader, _ = create_dataloader(test_data_file, batch_size, dataset_type, shuffle=False, encoder=encoder)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'encoder': encoder
    }

    return dataloaders


if __name__ == "__main__":
    data_dir = 'Gesture2Manuever_Prediction/GestureClassification/ExplicitGestureDataset'
    create_and_save_dataloaders(data_dir, 'explicit', 1)
    data_dir = 'Gesture2Manuever_Prediction/GestureClassification/ImplicitGestureDataset'
    create_and_save_dataloaders(data_dir, 'implicit', 1)
