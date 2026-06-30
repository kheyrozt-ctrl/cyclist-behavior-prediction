import pandas as pd
import os
from collections import Counter
import pickle

# 定义函数用于获取手势类型
def get_gesture_label(timestamp, ranges, labels, default_label):
    for (start, end), label in zip(ranges, labels):
        if start <= timestamp <= end:
            return label
    return default_label

def CropManueverData(gesture_info_path,skeleton_data_folder,output_file):
    # 读取 gesture_crop_info.csv
    gesture_info = pd.read_csv(gesture_info_path)




    # 定义 explicit 和 implicit gesture 的类型
    explicit_gestures = ['Left_Hand_Explicit', 'Right_Hand_Explicit']
    implicit_gestures = ['Left_Look', 'Left_Overshoulder', 'Right_Look']

    # 定义存储所有 segments 的列表
    all_data = []

    # 遍历每个文件名
    for filename in gesture_info['Filename'].unique():
        # 获取当前文件的相关 gesture 信息
        gestures = gesture_info[gesture_info['Filename'] == filename]

        # 判断 manuever_type 根据文件名的开头
        if filename.startswith(('video1a', 'video1b', 'video1g')):
            manuever_type = 'LeftTurn'
        elif filename.startswith('video1c'):
            manuever_type = 'RightTurn'
        elif filename.startswith(('video1d', 'video1e')):
            manuever_type = 'Crossing'
        else:
            break

        # print(f'processing file{filename}')

        # 读取对应的骨骼数据文件
        skeleton_file = os.path.join(skeleton_data_folder, filename)
        if not os.path.exists(skeleton_file):
            print(f"File {skeleton_file} not found, skipping.")
            continue

        # 读取骨骼数据并取最后749行
        skeleton_data = pd.read_csv(skeleton_file)
        if len(skeleton_data) < 749:
            print(f"File {filename} has less than 749 frames, skipping.")
            continue
        # skeleton_data = skeleton_data.tail(749)  # 仅保留最后749行

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

        for start_idx in range(30, len(skeleton_data) - 719 + 1):  # 滑动窗口从第30行开始
            data_matrix = []
            window_data = skeleton_data.iloc[start_idx:start_idx + 719]  # 取出窗口内的719行数据

            # 构建包含 explicit 和 implicit 标签的矩阵，维度为719×2
            data_matrix = []

            # 遍历骨骼数据的每个时间戳
            for _, row in window_data.iterrows():
                timestamp = row['timestamp']
                
                # 获取 explicit 和 implicit 标签
                explicit_label = get_gesture_label(timestamp, explicit_ranges, explicit_labels, 'neutral_explicit')
                implicit_label = get_gesture_label(timestamp, implicit_ranges, implicit_labels, 'neutral_implicit')

                # 添加到数据矩阵中
                if explicit_label == 'Left_Hand_Explicit':
                    explicit_num = 0
                elif explicit_label == 'Right_Hand_Explicit':
                    explicit_num = 1
                elif explicit_label == 'neutral_explicit':
                    explicit_num = 2
                if implicit_label == 'Left_Look':
                    implicit_num = 0
                elif implicit_label == 'Left_Overshoulder':
                    implicit_num = 1
                elif implicit_label == 'Right_Look':
                    implicit_num = 2
                elif implicit_label == 'neutral_implicit':
                    implicit_num = 3
                data_matrix.append([explicit_num, implicit_num])

            # 确保数据矩阵的维度是 719×2
            if len(data_matrix) == 719:
                all_data.extend([(data_matrix, manuever_type)])
            else:
                print(f"Data from {filename} did not reach 719 frames, skipping.")

    # 将数据保存为适当的格式（例如pkl或其他）
    print(f"Total datasets generated: {len(all_data)}")

    # 统计每种gesture_type的数量
    manuever_counts = Counter([manuever_type for _, manuever_type in all_data])

    # 打印统计结果
    print("Manuever Type Counts:")
    for manuever_type, count in manuever_counts.items():
        print(f"{manuever_type}: {count} segments")

    

    # 保存 all_segments 到文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)

    print(f"All segments saved to {output_file}")
