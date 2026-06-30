# dataloader.py
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def ManueverLoaddata(input_file,device,outputfile):
    with open(input_file, 'rb') as f:
        all_segments = pickle.load(f)

    # 1. 准备数据
    X = []
    y = []

    # 获取每个 segment 的固定形状（假设所有段应该有相同的形状）
    sample_segment = np.array(all_segments[0][0])
    expected_shape = sample_segment.shape

    # 遍历 all_segments，分离出特征和标签
    for segment, manuever_type in all_segments:
       
        segment_array = np.array(segment)  # 将每个 segment 转换为 NumPy 数组

        # 确保所有段的形状一致（通过填充或截断）
        if segment_array.shape != expected_shape:
            padded_segment = np.zeros(expected_shape)
            min_shape = np.minimum(segment_array.shape, expected_shape)
            padded_segment[:min_shape[0], :min_shape[1]] = segment_array.values[:min_shape[0], :min_shape[1]]
        else:
            padded_segment = segment_array
        
        
        
        X.append(padded_segment)
        y.append(manuever_type)
        

        

    # 将X和y转换为numpy数组
    X = np.array(X)
    y = np.array(y)



    # 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


    # 划分数据集：70%训练集，20%验证集，10%测试集
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    

    # 打印训练集、验证集和测试集的大小
    print(f"Training + Validation data size: {X_temp.shape[0]} samples")
    print(f"Testing data size: {X_test.shape[0]} samples")


    # 转换为Tensor并移动到GPU（如果有）
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)


    # 创建测试集DataLoader
    test_dataset = TensorDataset(X_test, y_test)


    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    


    # 打印 DataLoader 中的批次数量
    print(f"Number of testing batches: {len(test_loader)}")


    # 保存DataLoader相关的变量以供下个脚本使用
    with open(outputfile, 'wb') as f:
        pickle.dump((X_temp, y_temp, test_loader, device, label_encoder, expected_shape, len(np.unique(y))), f)
    