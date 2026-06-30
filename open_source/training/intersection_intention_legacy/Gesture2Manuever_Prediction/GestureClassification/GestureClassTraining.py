
# import sys
# sys.path.append("Gesture2Manuever_Prediction")
# from VideoCrop.GestureCrop import CropGestureData
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# checking if GPU is available
device = torch.device("cpu")

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# 加载之前保存的 all_segments
input_file = 'Gesture2Manuever_Prediction/VideoCrop/all_segments.pkl'

with open(input_file, 'rb') as f:
    all_segments = pickle.load(f)

# 1. 准备数据
X = []
y = []

# 获取每个 segment 的固定形状（假设所有段应该有相同的形状）
sample_segment = all_segments[0][0].drop(columns=['timestamp'])
expected_shape = sample_segment.shape

# 遍历 all_segments，分离出特征和标签
for segment, gesture_type in all_segments:
    # 去除 timestamp 列（假设第一列是 timestamp）
    segment_without_timestamp = segment.drop(columns=['timestamp'])
    
    # 确保所有段的形状一致（通过填充或截断）
    if segment_without_timestamp.shape != expected_shape:
        padded_segment = np.zeros(expected_shape)
        min_shape = np.minimum(segment_without_timestamp.shape, expected_shape)
        padded_segment[:min_shape[0], :min_shape[1]] = segment_without_timestamp.values[:min_shape[0], :min_shape[1]]
    else:
        padded_segment = segment_without_timestamp.values
    
    # 保持原始的2D形状
    X.append(padded_segment)
    y.append(gesture_type)

# 将X和y转换为numpy数组
X = np.array(X)
y = np.array(y)

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集和测试集的大小
print(f"Training data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 打印 DataLoader 中的批次数量
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

# 模型参数
input_size = expected_shape[1]
hidden_size = 128
num_layers = 2
num_classes = len(np.unique(y))

# 实例化模型
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
num_epochs = 200
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


    # 验证阶段
    model.eval()
    running_val_loss = 0.0
    # y_pred = []
    # y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            # y_pred.extend(predicted.cpu().numpy())
            # y_true.extend(labels.cpu().numpy())
    
    avg_val_loss = running_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
        
    # 打印每个 epoch 的训练损失和验证损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
    
    # scheduler.step()

# 打印Accuracy
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在GPU上
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 训练完成后保存模型
model_save_path = 'Gesture2Manuever_Prediction/GestureClassification/Gesture_LSTM_Model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
# 保存 LabelEncoder
LabelEncoder_save_path = 'Gesture2Manuever_Prediction/GestureClassification/label_encoder.pkl'
with open(LabelEncoder_save_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"LabelEncoder saved to {LabelEncoder_save_path}")


# 绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=0)
plt.savefig('Gesture2Manuever_Prediction/GestureClassification/Gesture_LSTM_Model.jpg')
plt.show()
