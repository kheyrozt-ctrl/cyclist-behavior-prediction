import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# 读取CSV文件
file_path_0 = "./Output_Data/sample_a_skeleton_keypoints.csv"
file_path_2 = "./Output_Data/sample_b_skeleton_keypoints.csv"

df_0 = pd.read_csv(file_path_0)
df_2 = pd.read_csv(file_path_2)

# 从第二行开始，并省去第一列
data_0 = df_0.iloc[1:, 1:].values  # 跳过第一行，并去掉第一列
data_2 = df_2.iloc[1:, 1:].values  # 跳过第一行，并去掉第一列

batch_size = 15

# 分组打包数据，每15行作为一组
sequences_0 = np.array([data_0[i * batch_size:(i + 1) * batch_size] for i in range(len(data_0) // batch_size)])
sequences_2 = np.array([data_2[i * batch_size:(i + 1) * batch_size] for i in range(len(data_2) // batch_size)])

# 创建标签
labels_0 = np.zeros(len(sequences_0), dtype=int)  # 标签全为0
labels_2 = np.full(len(sequences_2), 1, dtype=int)  # 标签全为1

# 合并数据和标签
sequences = np.concatenate((sequences_0, sequences_2), axis=0)
labels = np.concatenate((labels_0, labels_2), axis=0)
n_classes = 2

# 自定义 Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 数据集划分
dataset = SequenceDataset(sequences, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=3, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects input shape: (S, N, E)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Global average pooling over sequence dimension
        output = self.fc(output)
        return output
    
# 初始化模型
input_dim = sequences.shape[2]  # 每一行的特征数
model = TransformerModel(input_dim=input_dim, num_classes=n_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# 测试模型
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

# 运行训练和测试
train(model, train_loader, criterion, optimizer, epochs=10)
test(model, test_loader)

# 打印模型结构
print(model)
