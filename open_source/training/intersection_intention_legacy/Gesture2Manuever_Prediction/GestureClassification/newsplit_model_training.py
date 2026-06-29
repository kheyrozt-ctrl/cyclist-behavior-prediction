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
from GestureClassification.newsplit_exmodel_definition import ExplicitLSTMClassifier
from GestureClassification.newsplit_immodel_definition import ImplicitLSTMClassifier


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Train a single fold with early stopping based on validation loss."""
    patience = 40
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_f1 = 0.0
    best_acc = 0.0
    best_model_state = None
    best_model_epoch = 0
    patience_counter = 0

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

        # Save model if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_f1 = f1_val
            best_acc = acc_val * 100
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            best_model_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early Stopping triggered at epoch {epoch}")
                break

        scheduler.step()

    return train_losses, val_losses, best_model_state, best_model_epoch, best_acc, best_f1, best_val_loss


def get_input_output_size(dataloader):
    """Get input feature size and number of output classes from a dataloader."""
    batch = next(iter(dataloader))
    X, y = batch
    input_size = X.shape[2]
    output_size = len(torch.unique(y))
    return input_size, output_size


def train_all_folds(dataloaders, model_save_dir, LSTMClassifier, fold_number,
                    num_epochs=80, hidden_size=128, learning_rate=0.001, num_layers=4,
                    patience=5, eval_period=2):
    """Train gesture classification model for a single fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    encoder = dataloaders['encoder']

    input_size, output_size = get_input_output_size(train_loader)

    # Compute class weights from training data for imbalanced classes
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)
    class_counts = torch.bincount(all_labels, minlength=output_size).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"Training model for fold {fold_number}...")

    train_losses, val_losses, best_model_state, best_model_epoch, best_acc, best_f1, best_val_loss = \
        train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(train_losses) + 1), val_losses, label='Val Loss')
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
          f"with Val Loss: {best_val_loss:.4f}, Val Acc: {best_acc:.4f}, Val f1: {best_f1:.4f}.")

    # Save DataLoader and encoder for downstream use
    dataloader_save_path = os.path.join(model_save_dir, f'dataloader_fold_{fold_number}.pkl')
    with open(dataloader_save_path, 'wb') as f:
        pickle.dump((train_loader, val_loader, test_loader, encoder), f)

    # Evaluate best model on test set
    best_model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    y_pred = []
    y_true = []

    total_inference_time = 0
    num_samples = 0

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            single_start_time = time.time()
            outputs = best_model(inputs)
            single_end_time = time.time()
            total_inference_time += single_end_time - single_start_time
            num_samples += inputs.shape[0]
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    average_inference_time = total_inference_time / num_samples
    print(f'Average inference time per input: {average_inference_time:.6f} seconds')

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Train Accuracy: {100 * correct / total:.2f}%')

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Val Accuracy: {100 * correct / total:.2f}%')

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Testset')
    plt.xticks(rotation=0)
    confusion_matrix_save_path = os.path.join(model_save_dir, f'best_model_confusion_matrix_fold_{fold_number}.png')
    plt.savefig(confusion_matrix_save_path)
    plt.close()
