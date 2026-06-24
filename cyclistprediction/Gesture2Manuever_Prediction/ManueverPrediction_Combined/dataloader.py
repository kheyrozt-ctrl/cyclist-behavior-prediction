import os
import sys
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Compatibility shim: pickle files created with numpy >= 2.0 reference numpy._core
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray


def create_dataloader(data_file, batch_size, shuffle=True, label_encoder=None):
    """Load data from pickle file and create a DataLoader.

    If label_encoder is provided, uses transform() to ensure consistent encoding.
    If None, fits a new encoder on this data (should only be done for training set).
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    X = [torch.tensor(x[0], dtype=torch.float32) for x in data]  # Features: 749x66 matrices
    y = [x[1] for x in data]  # Labels: maneuver type

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)

    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    dataset = TensorDataset(torch.stack(X), y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, label_encoder


def create_and_save_dataloaders(data_dir, fold_number, batch_size=16):
    """Create DataLoaders for a given fold.

    Fits LabelEncoder on training data only, then uses transform() for val/test
    to ensure consistent label encoding across all splits.
    """
    train_data_file = os.path.join(data_dir, f'fold_{fold_number}_train_data.pkl')
    val_data_file = os.path.join(data_dir, f'fold_{fold_number}_val_data.pkl')
    test_data_file = os.path.join(data_dir, f'fold_{fold_number}_test_data.pkl')

    print(f"Creating DataLoader for fold {fold_number}...")
    # Fit encoder on training data only
    train_loader, label_encoder = create_dataloader(train_data_file, batch_size)
    # Use the same encoder for val/test to ensure consistent label mapping
    val_loader, _ = create_dataloader(val_data_file, batch_size, shuffle=False, label_encoder=label_encoder)
    test_loader, _ = create_dataloader(test_data_file, batch_size, shuffle=False, label_encoder=label_encoder)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'label_encoder': label_encoder
    }

    return dataloaders
