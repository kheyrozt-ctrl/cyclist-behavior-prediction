import pandas as pd
import os
import pickle
from collections import Counter

# 定义函数用于获取手势类型
def get_gesture_label(timestamp, ranges, labels, default_label):
    for (start, end), label in zip(ranges, labels):
        if start <= timestamp <= end:
            return label
    return default_label

# 函数：处理数据生成
def process_files(file_list, data, gesture_info, skeleton_data_folder):
    
    for filename in file_list:
        filename = filename.split('.', 1)[0]
        skeleton_file_name_begin = 'video' + filename.strip()
        
        # 获取当前文件的相关 gesture 信息
        gestures = gesture_info[gesture_info['Filename'].str.startswith(skeleton_file_name_begin, na=False)]
        
        # 判断 manuever_type 根据文件名的开头
        if filename.startswith(('1a', '1b', '1g')):
            manuever_type = 'LeftTurn'
        elif filename.startswith('1c'):
            manuever_type = 'RightTurn'
        elif filename.startswith(('1d', '1e')):
            manuever_type = 'Crossing'
        else:
            continue

        # 获取唯一的骨骼文件名
        if gestures.empty:
            print(f"No gestures found for {filename}, skipping.")
            continue

        skeleton_file_name = gestures['Filename'].unique()[0]  # 唯一与手势相关的文件名
        
        # 读取对应的骨骼数据文件
        skeleton_file = os.path.join(skeleton_data_folder, skeleton_file_name)
        if not os.path.exists(skeleton_file):
            print(f"File {skeleton_file} not found, skipping.")
            continue

        # 读取骨骼数据并取前749行，排除timestamp列，只保留66列
        skeleton_data = pd.read_csv(skeleton_file)
        if len(skeleton_data) < 749:
            print(f"File {filename} has less than 749 frames, skipping.")
            continue
        skeleton_data = skeleton_data.iloc[:, 1:67]  # 选择前749行和66列（排除timestamp）



        # 滑动窗口截取数据
        window_size = 749  # 窗口大小为749
        step_size = 3 if manuever_type == 'LeftTurn' else 1 if manuever_type == 'RightTurn' else 2


        for start_idx in range(0, len(skeleton_data) - window_size + 1, step_size):
            data_matrix = skeleton_data.iloc[start_idx:start_idx + window_size].to_numpy()

            # 添加到数据集中
            data.append((data_matrix, manuever_type))

    return data

def CropManueverData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number):
    # 读取 gesture_crop_info.csv
    gesture_info = pd.read_csv(gesture_info_path)

    
    # 定义存储所有 segments 的列表
    train_data, val_data, test_data = [], [], []

    # 处理训练、验证、测试数据
    train_data = process_files(train_files, train_data, gesture_info, skeleton_data_folder)
    val_data = process_files(val_files, val_data, gesture_info, skeleton_data_folder)
    test_data = process_files(test_files, test_data, gesture_info, skeleton_data_folder)

    # 统计每种manuever_type的数量
    with open(os.path.join(output_dir, f'fold_{fold_number}_data_Manuever_Type_Counts.txt'), 'w') as f:
        print("Manuever Type Counts:")
        f.write("Manuever Type Counts:\n")
        for dataset, name in [(train_data, "Training"), (val_data, "Validation"), (test_data, "Testing")]:
            manuever_counts = Counter([manuever_type for _, manuever_type in dataset])
            print(f"{name} Set:")
            f.write(f"{name}:\n")
            for manuever_type, count in manuever_counts.items():
                print(f"{manuever_type}: {count} segments")
                f.write(f"{manuever_type}: {count} segments\n")
    print(f"Manuever Type Counts saved to fold_{fold_number}_data_Manuever_Type_Counts.txt.")
    
    # 保存数据到文件
    with open(os.path.join(output_dir, f'fold_{fold_number}_train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, f'fold_{fold_number}_val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, f'fold_{fold_number}_test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Data saved to {output_dir} directory.")


# 示例用法
gesture_info_path = 'Gesture2Manuever_Prediction/VideoCrop/gesture_crop_info.csv'
skeleton_data_folder = 'Full33SkeletonData'
output_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/ManueverDataset'

for fold_number in range(1, 6):
    # 加载每个fold的文件列表
    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f]

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_val_files.txt', 'r') as f:
        val_files = [line.strip() for line in f]

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_test_files.txt', 'r') as f:
        test_files = [line.strip() for line in f]

    # 执行数据截取
    CropManueverData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number)
