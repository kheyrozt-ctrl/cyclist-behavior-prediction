import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.preprocessing import LabelEncoder

# Function to load data from pickle file and create a DataLoader
def create_dataloader(data_file, batch_size, shuffle=True):
    # Load data from pickle file
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Separate data into features (X) and labels (y)
    X = [torch.tensor(x[0], dtype=torch.float32) for x in data]  # Features: List of 719x2 matrices
    y = [x[1] for x in data]  # Labels: manuever types

    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # Create TensorDataset
    dataset = TensorDataset(torch.stack(X), y_tensor)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, label_encoder

# Main function to create DataLoaders for each fold and save them
def create_and_save_dataloaders(data_dir, output_dir, batch_size=16):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fold_number in range(1, 6):
        # Define paths for each fold's data
        train_data_file = os.path.join(data_dir, f'fold_{fold_number}_train_data.pkl')
        val_data_file = os.path.join(data_dir, f'fold_{fold_number}_val_data.pkl')
        test_data_file = os.path.join(data_dir, f'fold_{fold_number}_test_data.pkl')

        # Create DataLoaders for train, val, and test
        print(f"Creating DataLoader for fold {fold_number}...")
        train_loader, train_label_encoder = create_dataloader(train_data_file, batch_size)
        val_loader, val_label_encoder = create_dataloader(val_data_file, batch_size, shuffle=False)
        test_loader, test_label_encoder = create_dataloader(test_data_file, batch_size, shuffle=False)

        # Ensure all label encoders are the same
        assert train_label_encoder.classes_.tolist() == val_label_encoder.classes_.tolist() == test_label_encoder.classes_.tolist(), \
            "Label encoders for train, val, and test do not match."

        # Save DataLoaders and the label encoder
        dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'label_encoder': train_label_encoder
        }

        output_file = os.path.join(output_dir, f'dataloaders_fold_{fold_number}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(dataloaders, f)
        
        print(f"DataLoaders for fold {fold_number} saved to {output_file}.")

if __name__ == "__main__":
    # Example usage
    data_dir = 'Gesture2Manuever_Prediction/ManueverPrediction/ManueverDataset'  # Path to the folder containing the .pkl files
    output_dir = 'Gesture2Manuever_Prediction/ManueverPrediction/DataLoaders'
    create_and_save_dataloaders(data_dir, output_dir)
