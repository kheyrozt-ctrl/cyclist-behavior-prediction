import pandas as pd
import os
import random
from collections import Counter
import pickle

# 定义函数用于获取手势类型
def get_gesture_label(timestamp, ranges, labels, default_label):
    for (start, end), label in zip(ranges, labels):
        if start <= timestamp <= end:
            return label
    return default_label

# process_files(train_files, train_data, gesture_info, skeleton_data_folder, explicit_gestures, implicit_gestures)
# 函数：处理数据生成
def process_files(file_list, data, gesture_info, skeleton_data_folder, explicit_gestures, implicit_gestures):
    
    for filename in file_list:
        filename = filename.split('.',1)[0]
        skeleton_file_name_begin = 'video' + filename.strip()
        # 获取当前文件的相关 gesture 信息
        
        gestures = gesture_info[gesture_info['Filename'].str.startswith(skeleton_file_name_begin, na=False)]
        
        # gestures = gestures.dropna() #drop all rows that have any NaN values
        

        # 判断 manuever_type 根据文件名的开头
        if filename.startswith(('1a', '1b', '1g')):
            manuever_type = 'LeftTurn'
            slide_interval = 3
        elif filename.startswith('1c'):
            manuever_type = 'RightTurn'
            slide_interval = 1
        elif filename.startswith(('1d', '1e')):
            manuever_type = 'Crossing'
            slide_interval = 3
        else:
            break

        # Construct the skeleton file path from the matched entries in gesture_info
        print(gestures['Filename'].unique())
        skeleton_file_name = gestures['Filename'].unique()[0]  # Unique filenames associated with this gesture
        
        # 读取对应的骨骼数据文件
        skeleton_file = os.path.join(skeleton_data_folder, skeleton_file_name)
        if not os.path.exists(skeleton_file):
            print(f"File {skeleton_file} not found, skipping.")
            continue

        # 读取骨骼数据并取最后749行
        skeleton_data = pd.read_csv(skeleton_file)
        if len(skeleton_data) < 749:
            print(f"File {filename} has less than 749 frames, skipping.")
            continue

        # 初始化 explicit 和 implicit 手势的时间戳范围和标签
        explicit_ranges = []
        explicit_labels = []
        implicit_ranges = []
        implicit_labels = []

        # 遍历每个手势，获取时间戳范围并存储相应的手势标签
        for _, row in gestures.iterrows():
            gesture_type = row['Gesture Type']
            start_ts = row['Start Timestamp']
            end_ts = row['End Timestamp']

            # 分离 explicit 和 implicit 手势范围和标签
            if gesture_type in explicit_gestures:
                explicit_ranges.append((start_ts + 1.0, end_ts))
                explicit_labels.append(gesture_type)
            elif gesture_type in implicit_gestures:
                implicit_ranges.append((start_ts + 1.0, end_ts))
                implicit_labels.append(gesture_type)
        # print(len(skeleton_data))
        # 滑动窗口从第30行开始
        for start_idx in range(30, len(skeleton_data) - 719 + 1, slide_interval):
            data_matrix = []
            window_data = skeleton_data.iloc[start_idx:start_idx + 719]  # 取出窗口内的719行数据

            # 构建包含 explicit 和 implicit 标签的矩阵，维度为719×2
            for _, row in window_data.iterrows():
                timestamp = row['timestamp']
                
                # 获取 explicit 和 implicit 标签
                explicit_label = get_gesture_label(timestamp, explicit_ranges, explicit_labels, 'neutral_explicit')
                implicit_label = get_gesture_label(timestamp, implicit_ranges, implicit_labels, 'neutral_implicit')

                # 转换标签为数值形式
                explicit_num = {'Left_Hand_Explicit': -1, 'Right_Hand_Explicit': 1, 'neutral_explicit': 0}[explicit_label]
                implicit_num = {'Left_Look': -1, 'Left_Overshoulder': -2, 'Right_Look': 1, 'neutral_implicit': 0}[implicit_label]
                data_matrix.append([explicit_num, implicit_num])

            # 确保数据矩阵的维度是 719×2
            if len(data_matrix) == 719:
                data.extend([(data_matrix, manuever_type)])
                # print(data)
            else:
                print(f"Data from {filename} did not reach 719 frames, skipping.")
            
    return data    


def CropManueverData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number):
    # 读取 gesture_crop_info.csv
    gesture_info = pd.read_csv(gesture_info_path)

    # 定义 explicit 和 implicit gesture 的类型
    explicit_gestures = ['Left_Hand_Explicit', 'Right_Hand_Explicit']
    implicit_gestures = ['Left_Look', 'Left_Overshoulder', 'Right_Look']

    # 定义存储所有 segments 的列表
    train_data, val_data, test_data = [], [], []

    
    # 处理训练、验证、测试数据
    train_data = process_files(train_files, train_data, gesture_info, skeleton_data_folder, explicit_gestures, implicit_gestures)
    val_data = process_files(val_files, val_data, gesture_info, skeleton_data_folder, explicit_gestures, implicit_gestures)
    test_data = process_files(test_files, test_data, gesture_info, skeleton_data_folder, explicit_gestures, implicit_gestures)

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

    # # 保存Manuever Type Counts到文件
    # with open(os.path.join(output_dir, f'fold_{fold_number}_data_Manuever_Type_Counts.txt'), 'w') as f:
    #     f.write("Manuever Type Counts:\n")
    #     for dataset_name, counts in manuever_counts.items():
    #         f.write(f"{dataset_name}:\n")
    #         for manuever_type, count in counts.items():
    #             f.write(f"{manuever_type}: {count} segments\n")

    


# 示例用法
gesture_info_path = 'Gesture2Manuever_Prediction/VideoCrop/gesture_crop_info.csv'
skeleton_data_folder = 'Full33SkeletonData'
output_dir = 'Gesture2Manuever_Prediction/ManueverPrediction/ManueverDataset'

# Replace fold_number with the appropriate number for each fold

for fold_number in range(1,6):
    # Load the file lists for the current fold
    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f]

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_val_files.txt', 'r') as f:
        val_files = [line.strip() for line in f]

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_test_files.txt', 'r') as f:
        test_files = [line.strip() for line in f]

    # Run data cropping
    CropManueverData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number)
