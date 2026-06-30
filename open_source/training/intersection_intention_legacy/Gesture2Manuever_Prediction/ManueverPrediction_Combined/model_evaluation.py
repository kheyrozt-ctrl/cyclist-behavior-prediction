import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

sys.path.append("Gesture2Manuever_Prediction")
from ManueverPrediction_Combined.model_definition import CombinedModel
from GestureClassification.newsplit_exmodel_definition import ExplicitLSTMClassifier
from GestureClassification.newsplit_immodel_definition import ImplicitLSTMClassifier

def get_input_output_size(dataloader):
    batch = next(iter(dataloader))
    X, y = batch
    input_size = X.shape[2]
    output_size = len(torch.unique(y))
    return input_size, output_size

# Validation function
def validate_model(model, train_loader, val_loader, device, label_encoder):
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    acc_val = accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, average='weighted')
    print(f'Train+Validation Accuracy: {acc_val:.2f}, Train+Validation F1 Score: {f1_val:.2f}')
    
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Validation Set')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix_validation.png'))
    plt.close()
    
    return acc_val, f1_val

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/Models_ExplicitImplicit'
    best_model_path = os.path.join(model_save_dir, 'best_model_fold_5.pkl')
    dataloader_path = os.path.join(model_save_dir, 'dataloader_fold_5.pkl')
    gesture_model_explicit_path = 'Gesture2Manuever_Prediction/GestureClassification/ExplicitModels/best_model_fold_5.pkl'
    gesture_model_implicit_path = 'Gesture2Manuever_Prediction/GestureClassification/ImplicitModels/best_model_fold_5.pkl'
    gesture_model_explicit_dataloaders_path = 'Gesture2Manuever_Prediction/GestureClassification/ExplicitModels/dataloader_fold_5.pkl'
    gesture_model_implicit_dataloaders_path = 'Gesture2Manuever_Prediction/GestureClassification/ImplicitModels/dataloader_fold_5.pkl'
    
    # Load DataLoader
    with open(dataloader_path, 'rb') as f:
        train_loader, val_loader, _, device, label_encoder = pickle.load(f)
    
    # Load gesture recognition models
    with open(gesture_model_explicit_dataloaders_path, 'rb') as f:
        explicit_train_loader, _, _, label_encoder_explicit = pickle.load(f)
        input_size, num_classes = get_input_output_size(explicit_train_loader)
        gesture_model_explicit = ExplicitLSTMClassifier(input_size, 128, 4, num_classes)
        gesture_model_explicit.load_state_dict(torch.load(gesture_model_explicit_path))
    
    with open(gesture_model_implicit_dataloaders_path, 'rb') as f:
        implicit_train_loader, _, _, label_encoder_implicit = pickle.load(f)
        input_size, num_classes = get_input_output_size(implicit_train_loader)
        gesture_model_implicit = ImplicitLSTMClassifier(input_size, 128, 4, num_classes)
        gesture_model_implicit.load_state_dict(torch.load(gesture_model_implicit_path))
    
    # Define maneuver prediction model
    hidden_size = 128
    manuever_output_size = len(label_encoder.classes_)
    model = CombinedModel(gesture_model_explicit, gesture_model_implicit, label_encoder_explicit, label_encoder_implicit, hidden_size, manuever_output_size).to(device)
    
    # Load trained model
    model.load_state_dict(torch.load(best_model_path))
    
    # Run validation
    validate_model(model, train_loader, val_loader, device, label_encoder)
