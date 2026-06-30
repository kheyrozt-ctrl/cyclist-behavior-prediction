import os
import sys
import time
sys.path.append("Gesture2Manuever_Prediction")
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
from ManueverPrediction_Combined.model_definition import (
    CombinedModel, CombinedModel_Explicit,
    CombinedModel_HiddenWeighting, CombinedModel_HiddenWeighting_Explicit
)
from GestureClassification.newsplit_exmodel_definition import ExplicitLSTMClassifier
from GestureClassification.newsplit_immodel_definition import ImplicitLSTMClassifier


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Train a single fold, selecting model by best weighted F1 score."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_f1 = 0.0
    best_model_state = None
    best_model_epoch = 0

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        y_pred = []
        y_true = []
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                total += labels.size(0)
                loss = criterion(outputs, labels)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        acc_val = accuracy_score(y_true, y_pred)
        f1_val = f1_score(y_true, y_pred, average='weighted')

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {acc_val:.4f}, Val F1: {f1_val:.4f}")

        # Save model if F1 improved
        if f1_val > best_f1:
            best_f1 = f1_val
            best_model_state = model.state_dict()
            best_model_epoch = epoch

        scheduler.step()

    return train_losses, val_losses, best_model_state, best_model_epoch


def get_input_output_size(dataloader):
    """Get input feature size and number of output classes from a dataloader."""
    batch = next(iter(dataloader))
    X, y = batch
    input_size = X.shape[2]
    output_size = len(torch.unique(y))
    return input_size, output_size


def _create_combined_model(model_type, model_subtype, gesture_model_explicit, gesture_model_implicit,
                           label_encoder_explicit, label_encoder_implicit, hidden_size, output_size, device):
    """Create the appropriate combined model based on model_type and model_subtype."""
    model_classes = {
        ("weighting_NN", "Ex_Im"): CombinedModel_HiddenWeighting,
        ("weighting_NN", "Ex_only"): CombinedModel_HiddenWeighting_Explicit,
        ("weighting_set", "Ex_Im"): CombinedModel,
        ("weighting_set", "Ex_only"): CombinedModel_Explicit,
    }
    model_cls = model_classes[(model_type, model_subtype)]
    return model_cls(gesture_model_explicit, gesture_model_implicit,
                     label_encoder_explicit, label_encoder_implicit,
                     hidden_size, output_size).to(device)


def train_all_folds(model_type, model_subtype, dataloaders, model_save_dir,
                    gesture_model_explicit_dataloaders_path, gesture_model_implicit_dataloaders_path,
                    gesture_model_explicit_path, gesture_model_implicit_path, fold_number,
                    num_epochs=30, hidden_size=128, learning_rate=0.0001, num_layers=2,
                    patience=5, eval_period=5):
    """Train combined maneuver prediction model for a single fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Load pre-trained explicit gesture model
    with open(gesture_model_explicit_dataloaders_path, 'rb') as f:
        explicit_train_loader, _, _, label_encoder = pickle.load(f)
        input_size, num_classes = get_input_output_size(explicit_train_loader)
        gesture_hidden_size = 128
        gesture_num_layers = 4
        gesture_model_explicit = ExplicitLSTMClassifier(input_size, gesture_hidden_size, gesture_num_layers, num_classes)
        gesture_model_explicit.load_state_dict(torch.load(gesture_model_explicit_path))
        label_encoder_explicit = label_encoder

    # Load pre-trained implicit gesture model
    with open(gesture_model_implicit_dataloaders_path, 'rb') as f:
        implicit_train_loader, _, _, label_encoder = pickle.load(f)
        input_size, num_classes = get_input_output_size(implicit_train_loader)
        gesture_model_implicit = ImplicitLSTMClassifier(input_size, gesture_hidden_size, gesture_num_layers, num_classes)
        gesture_model_implicit.load_state_dict(torch.load(gesture_model_implicit_path))
        label_encoder_implicit = label_encoder

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    label_encoder = dataloaders['label_encoder']

    manuever_output_size = len(label_encoder.classes_)

    # Create model using helper (eliminates duplicated if/elif blocks)
    model = _create_combined_model(model_type, model_subtype, gesture_model_explicit,
                                   gesture_model_implicit, label_encoder_explicit,
                                   label_encoder_implicit, hidden_size, manuever_output_size, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"Training model for fold {fold_number}...")

    train_losses, val_losses, best_model_state, best_model_epoch = \
        train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, best_model_epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, best_model_epoch + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_number} - Training and Validation Loss')
    plt.legend()
    loss_plot_save_path = os.path.join(model_save_dir, f'best_model_loss_plot_{fold_number}.png')
    plt.savefig(loss_plot_save_path)
    plt.close()

    # Save the best model
    model_save_path = os.path.join(model_save_dir, f'best_model_fold_{fold_number}.pkl')
    torch.save(best_model_state, model_save_path)
    print(f"Best model for fold {fold_number} saved at epoch {best_model_epoch} "
          f"with Val Loss: {val_losses[best_model_epoch - 1]:.4f}.")

    # Save DataLoader variables for downstream use
    dataloader_save_path = os.path.join(model_save_dir, f'dataloader_fold_{fold_number}.pkl')
    with open(dataloader_save_path, 'wb') as f:
        pickle.dump((train_loader, val_loader, test_loader, device, label_encoder), f)

    # Evaluate best model on test set
    print(f"Evaluating on test set with best validation loss: {val_losses[best_model_epoch - 1]:.4f}")
    best_model = _create_combined_model(model_type, model_subtype, gesture_model_explicit,
                                        gesture_model_implicit, label_encoder_explicit,
                                        label_encoder_implicit, hidden_size, manuever_output_size, device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    total_inference_time = 0
    num_samples = 0
    y_pred_test = []
    y_true_test = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            single_start_time = time.time()
            outputs = best_model(inputs)
            single_end_time = time.time()
            total_inference_time += single_end_time - single_start_time
            num_samples += inputs.shape[0]
            _, predicted = torch.max(outputs.data, 1)
            y_pred_test.extend(predicted.cpu().numpy())
            y_true_test.extend(labels.cpu().numpy())

    acc_test = accuracy_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test, average='weighted')
    print(f'Test Accuracy: {acc_test:.2f}%, Test F1: {f1_test:.2f}')

    average_inference_time = total_inference_time / num_samples
    print(f'Average inference time per input: {average_inference_time:.6f} seconds')
    inference_time_save_path = os.path.join(model_save_dir, f'inference_time_fold_{fold_number}.txt')
    with open(inference_time_save_path, 'w') as f:
        f.write(f"Average inference time per input: {average_inference_time:.6f} seconds\n")

    # Evaluate on train set
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    acc_train = accuracy_score(y_true, y_pred)
    f1_train = f1_score(y_true, y_pred, average='weighted')
    print(f'Train Accuracy: {acc_train:.2f}%, Train F1: {f1_train:.2f}')

    # Evaluate on val set
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    acc_val = accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, average='weighted')
    print(f'Val Accuracy: {acc_val:.2f}%, Val F1: {f1_val:.2f}')

    # Plot confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Testset')
    plt.xticks(rotation=0)
    confusion_matrix_save_path = os.path.join(model_save_dir, f'best_model_confusion_matrix_fold_{fold_number}.png')
    plt.savefig(confusion_matrix_save_path)
    plt.close()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time for all folds: {total_time:.2f} seconds")
    training_time_save_path = os.path.join(model_save_dir, 'training_time.txt')
    with open(training_time_save_path, 'w') as f:
        f.write(f"Total training time for all folds: {total_time:.2f} seconds")
