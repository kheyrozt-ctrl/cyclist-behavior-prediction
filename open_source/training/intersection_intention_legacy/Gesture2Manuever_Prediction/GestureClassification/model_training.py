# model_training.py
import os
import pickle
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from GestureClassification.model_definition import LSTMClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def GestureTraining(dataloader_path,model_save_folder,LabelEncoder_save_folder,confusion_matrix_save_folder,num_epochs):
    # 加载DataLoader和其他相关变量
    with open(dataloader_path, 'rb') as f:
        X_temp, y_temp, test_loader, device, label_encoder, expected_shape, num_classes = pickle.load(f)

    # 模型参数
    input_size = expected_shape[1]
    hidden_size = 128
    num_layers = 2

    

    # 设置分层 K 折交叉验证
    k_folds = 9
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold = 1

    

    # 保存每个折的准确率
    fold_accuracies = []

    for train_idx, val_idx in kfold.split(X_temp, y_temp):

        print(f"Training fold {fold}...")
        best_val_loss = float('inf')
        best_model_state = None

        # 获取当前折的训练和验证集
        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        # 转换为Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

        # 创建DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # 实例化模型并移动到GPU（如果有）
        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

        # 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # 训练模型
        # num_epochs = 40
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            running_train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录训练损失
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)


            # # 验证阶段
            # model.eval()
            # running_val_loss = 0.0
            # with torch.no_grad():
            #     for inputs, labels in val_loader:
            #         inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
            #         outputs = model(inputs)
            #         loss = criterion(outputs, labels)
            #         running_val_loss += loss.item()

            
            # avg_val_loss = running_val_loss / len(test_loader)
            # val_losses.append(avg_val_loss)
                
            # # 打印每个 epoch 的训练损失和验证损失
            # print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


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

            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {acc_val:.4f}, Val F1: {f1_val:.4f}")

            # # 如果验证集的损失小于0.2，终止训练
            # if avg_val_loss < 0.2:
            #     print(f"Early stopping at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")
            #     break
            
            # scheduler.step()

            # 如果验证集损失低于当前最佳损失，保存模型
            if epoch_val_loss < best_val_loss:
                best_model_epoch = epoch
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict()
                best_train_dataloader = train_loader
                best_val_dataloader = val_loader
            
        
        print(f"Best model for fold {fold} saved at epoch {best_model_epoch} with Val Loss: {best_val_loss:.4f}.")

        # 保存DataLoader相关的变量以供下个脚本使用

        # with open(best_model_dataloader_save_file, 'wb') as f:
        #     pickle.dump((best_train_dataloader, best_val_dataloader, test_loader, device, label_encoder, expected_shape, num_classes), f)


        # 使用最佳模型在测试集上进行评估
        best_model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
        best_model.load_state_dict(best_model_state)
        best_model.eval()
        y_pred_test = []
        y_true_test = []
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
                outputs = best_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred_test.extend(predicted.cpu().numpy())
                y_true_test.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc_test = accuracy_score(y_true_test, y_pred_test)
            f1_test = f1_score(y_true_test, y_pred_test, average='weighted')
            print(f'Test Accuracy: {acc_test:.2f}%, Test F1: {f1_test:.2f}')
        
        y_pred = []
        y_true = []
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
                outputs = best_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc_train = accuracy_score(y_true, y_pred)
            f1_train = f1_score(y_true, y_pred, average='weighted')
            print(f'Train Accuracy: {acc_train:.2f}%, Train F1: {f1_train:.2f}')
        
        y_pred = []
        y_true = []
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
                outputs = best_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc_val = accuracy_score(y_true, y_pred)
            f1_val = f1_score(y_true, y_pred, average='weighted')
            print(f'Val Accuracy: {acc_val:.2f}%, Val F1: {f1_val:.2f}')


        # 训练完成后保存模型
        model_save_folder_ = os.path.join(model_save_folder,f'fold{fold}/')
        if not os.path.exists(model_save_folder_):
            os.makedirs(model_save_folder_)
        model_save_path = os.path.join(model_save_folder_,f'best_model_fold_{fold}.pkl')
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        # 保存 LabelEncoder
        LabelEncoder_save_folder_ = os.path.join(LabelEncoder_save_folder,f'fold{fold}/')
        if not os.path.exists(LabelEncoder_save_folder_):
            os.makedirs(LabelEncoder_save_folder_)
        LabelEncoder_save_path = os.path.join(LabelEncoder_save_folder,f'LabelEncoder_fold_{fold}.pkl')
        with open(LabelEncoder_save_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"LabelEncoder saved to {LabelEncoder_save_path}")


        # 绘制混淆矩阵
        confusion_matrix_save_folder_ = os.path.join(confusion_matrix_save_folder,f'fold{fold}/')
        if not os.path.exists(confusion_matrix_save_folder_):
            os.makedirs(confusion_matrix_save_folder_)
        confusion_matrix_save_path = os.path.join(confusion_matrix_save_folder_,f'best_model_CM_fold_{fold}.png')
        cm = confusion_matrix(y_true_test, y_pred_test)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix on Testset')
        plt.xticks(rotation=0)
        plt.savefig(confusion_matrix_save_path)
        plt.show()


        fold += 1

    
    