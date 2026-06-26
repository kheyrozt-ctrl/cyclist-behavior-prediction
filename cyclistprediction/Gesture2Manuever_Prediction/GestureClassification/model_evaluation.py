import pickle
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from GestureClassification.model_definition import LSTMClassifier

def AccCalculation(device, dataloader, model):
    y_pred = []
    y_true = []
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc_test = accuracy_score(y_true, y_pred)
    f1_test = f1_score(y_true, y_pred, average='weighted')

    return  acc_test, f1_test, y_pred, y_true

def GestureEvaluation(model_load_path,Bestmodel_dataloader_path,confusion_matrix_save_path):
    # 加载DataLoader和其他相关变量
    with open(Bestmodel_dataloader_path, 'rb') as f:
        best_train_dataloader, best_val_dataloader, test_loader, device, label_encoder, expected_shape, num_classes = pickle.load(f)

    # 模型参数
    input_size = expected_shape[1]
    hidden_size = 128
    num_layers = 2

    # 实例化模型并加载状态字典
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_load_path))

    # 将模型移动到GPU（如果可用）
    model.to(device)
    model.eval()  # 设置模型为评估模式


    acc_test, f1_test, y_pred_test, y_true_test = AccCalculation(device, test_loader, model)
    print(f'Test Accuracy: {acc_test:.2f}%, F1 Score: {f1_test:.2%}')

    acc_train, f1_train, _, _ = AccCalculation(device, best_train_dataloader, model)
    print(f'Train Accuracy: {acc_train:.2f}%, F1 Score: {f1_train:.2%}')

    acc_val, f1_val, _, _ = AccCalculation(device, best_val_dataloader, model)
    print(f'Validation Accuracy: {acc_val:.2f}%, F1 Score: {f1_val:.2%}')




    # 绘制混淆矩阵
    cm = confusion_matrix(y_true_test, y_pred_test)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix on Testset (Accuracy: {acc_test:.2%}, F1 Score: {f1_test:.2%})')
    plt.xticks(rotation=0)
    plt.savefig(confusion_matrix_save_path)
    plt.show()

