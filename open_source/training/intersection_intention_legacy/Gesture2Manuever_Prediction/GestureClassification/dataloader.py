# dataloader.py
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def GestureLoaddata(input_file,device,explicit_outputfile,implicit_outputfile):
    with open(input_file, 'rb') as f:
        all_segments = pickle.load(f)

    # 1. 准备数据
    X_explicit = []
    y_explicit = []
    X_implicit = []
    y_implicit = []

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
        

        # 将数据分为explicit和implicit
        if gesture_type in ['Left_Hand_Explicit', 'Right_Hand_Explicit', 'neutral_explicit']:
            X_explicit.append(padded_segment)
            y_explicit.append(gesture_type)
        else:
            X_implicit.append(padded_segment)
            y_implicit.append(gesture_type)

    # 将X和y转换为numpy数组
    X_explicit = np.array(X_explicit)
    y_explicit = np.array(y_explicit)
    X_implicit = np.array(X_implicit)
    y_implicit = np.array(y_implicit)


    # 标签编码
    explicit_label_encoder = LabelEncoder()
    y_explicit = explicit_label_encoder.fit_transform(y_explicit)

    implicit_label_encoder = LabelEncoder()
    y_implicit = implicit_label_encoder.fit_transform(y_implicit)

    # 划分数据集：70%训练集，20%验证集，10%测试集
    X_explicit_temp, X_explicit_test, y_explicit_temp, y_explicit_test = train_test_split(X_explicit, y_explicit, test_size=0.1, random_state=42, stratify=y_explicit)
    X_implicit_temp, X_implicit_test, y_implicit_temp, y_implicit_test = train_test_split(X_implicit, y_implicit, test_size=0.1, random_state=42, stratify=y_implicit)
    

    # 打印训练集、验证集和测试集的大小
    print(f"Explicit Training + Validation data size: {X_explicit_temp.shape[0]} samples")
    print(f"Explicit Testing data size: {X_explicit_test.shape[0]} samples")
    print(f"Implicit Training + Validation data size: {X_implicit_temp.shape[0]} samples")
    print(f"Implicit Testing data size: {X_implicit_test.shape[0]} samples")

    # 转换为Tensor并移动到GPU（如果有）
    X_explicit_test = torch.tensor(X_explicit_test, dtype=torch.float32).to(device)
    y_explicit_test = torch.tensor(y_explicit_test, dtype=torch.long).to(device)
    X_implicit_test = torch.tensor(X_implicit_test, dtype=torch.float32).to(device)
    y_implicit_test = torch.tensor(y_implicit_test, dtype=torch.long).to(device)

    # 创建测试集DataLoader
    explicit_test_dataset = TensorDataset(X_explicit_test, y_explicit_test)
    implicit_test_dataset = TensorDataset(X_implicit_test, y_implicit_test)

    explicit_test_loader = DataLoader(explicit_test_dataset, batch_size=64, shuffle=False)
    implicit_test_loader = DataLoader(implicit_test_dataset, batch_size=64, shuffle=False)


    # 打印 DataLoader 中的批次数量
    print(f"Number of explicit testing batches: {len(explicit_test_loader)}")
    print(f"Number of implicit testing batches: {len(implicit_test_loader)}")

    # 保存DataLoader相关的变量以供下个脚本使用
    with open(explicit_outputfile, 'wb') as f:
        pickle.dump((X_explicit_temp, y_explicit_temp, explicit_test_loader, device, explicit_label_encoder, expected_shape, len(np.unique(y_explicit))), f)
    with open(implicit_outputfile, 'wb') as f:
        pickle.dump((X_implicit_temp, y_implicit_temp, implicit_test_loader, device, implicit_label_encoder, expected_shape, len(np.unique(y_implicit))), f)
