import pandas as pd
import os
import pickle
from collections import Counter
import random
import gc

# Function to balance the gesture data
def balance_gesture_data(test_data, target_gesture='Left_Overshoulder'):
    # Count the occurrences of each gesture type
    gesture_counts = Counter([gesture_type for _, gesture_type in test_data])
    
    # Get the number of 'Left_Overshoulder' gestures
    target_count = gesture_counts.get(target_gesture, 0)
    
    # If there are no 'Left_Overshoulder' gestures, return the original data
    if target_count == 0:
        print(f"No '{target_gesture}' gestures found in test data.")
        return test_data

    balanced_test_data = []

    # Go through each gesture type and limit its occurrences to target_count
    for gesture_type, count in gesture_counts.items():
        gesture_data = [(seg, gesture_type) for seg, gtype in test_data if gtype == gesture_type]

        if gesture_type != target_gesture and count > target_count:
            # Randomly select target_count samples from the data
            gesture_data = random.sample(gesture_data, target_count)

        balanced_test_data.extend(gesture_data)

    return balanced_test_data

# 处理骨架文件，生成手势片段
def process_skeleton_file(filename, start_ts, end_ts, gesture_type, skeleton_data_folder, fold_number, datasettype, gesture_definition):
    skeleton_file_path = os.path.join(skeleton_data_folder, filename)
    skeleton_data = pd.read_csv(skeleton_file_path)

    cropped_data = skeleton_data[(skeleton_data['timestamp'] >= start_ts) & (skeleton_data['timestamp'] < end_ts)]

    # 滑动窗口采样
    segment_length = 1.0  # 每段数据为1.0秒
    slide_interval = None  # 初始化为None，确保每种情况下都会被正确赋值

    if fold_number == 1:
        if datasettype == 'train':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 3
        elif datasettype == 'val':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 1
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2
        else:
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2

    elif fold_number == 2:
        if datasettype == 'train':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 3
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2
        elif datasettype == 'val':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 1
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 5
        else:
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2

    elif fold_number == 3:
        if datasettype == 'train':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 1
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 3
        elif datasettype == 'val':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 1
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2
        else:
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2

    elif fold_number == 4:
        if datasettype == 'train':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 3
        elif datasettype == 'val':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 1
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 4
        else:
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2

    elif fold_number == 5:
        if datasettype == 'train':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 3
        elif datasettype == 'val':
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 1
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2
        else:
            if gesture_type == 'Left_Overshoulder':
                slide_interval = 1
            elif gesture_type == 'Left_Look':
                slide_interval = 1
            elif gesture_type == 'Right_Hand_Explicit':
                slide_interval = 1
            elif gesture_type == 'Right_Look':
                slide_interval = 2
            elif gesture_type == 'Left_Hand_Explicit':
                slide_interval = 2

    # 如果 slide_interval 仍然为 None，意味着该 gesture_type 未定义正确的处理，返回跳过
    if slide_interval is None:
        return [], 'Skip'
    
    if gesture_definition == 'explicit':
        if gesture_type in ['Right_Hand_Explicit', 'Left_Hand_Explicit']:
            segments = []
            if len(cropped_data) > segment_length * 30:  # 30Hz的采样率
                for start in range(0, len(cropped_data) - int(segment_length * 30), slide_interval):
                    segment = cropped_data.iloc[start:start + int(segment_length * 30),1:67]
                    segments.append(segment)
            elif len(cropped_data) == segment_length * 30:
                segments.append(cropped_data.iloc[:,1:67])
            return segments, gesture_type
        else:
            return [], 'Skip'
    elif gesture_definition == 'implicit':
        if gesture_type in ['Left_Look', 'Left_Overshoulder', 'Right_Look']:
            segments = []
            if len(cropped_data) > segment_length * 30:  # 30Hz的采样率
                for start in range(0, len(cropped_data) - int(segment_length * 30), slide_interval):
                    segment = cropped_data.iloc[start:start + int(segment_length * 30),1:67]
                    segments.append(segment)
            elif len(cropped_data) == segment_length * 30:
                segments.append(cropped_data.iloc[:,1:67])
            return segments, gesture_type
        else:
            return [], 'Skip'

    


# 获取中立手势的时间段，排除标注手势前后 1.0 秒
def get_neutral_intervals(start_ts_list, end_ts_list, total_duration=24.0):
    neutral_intervals = []
    current_start = 1.0

    for start_ts, end_ts in sorted(zip(start_ts_list, end_ts_list)):
        neutral_end = start_ts - 1.0
        if neutral_end > current_start:
            neutral_intervals.append((current_start, min(neutral_end, total_duration)))
        current_start = end_ts + 1.0

    if current_start < total_duration - 1.0:
        neutral_intervals.append((current_start, total_duration))

    return neutral_intervals


# 处理中立手势片段
def process_neutral_segments(filename_prefix, neutral_intervals, skeleton_data_folder, fold_number, datasettype, gesture_definition, segment_length=1.0, slide_interval=53):
    if gesture_definition == 'explicit':
        if fold_number == 1:
            if datasettype == 'train':
                slide_interval = 10
            elif datasettype == 'val':
                slide_interval = 12
            else:
                slide_interval = 12
        elif fold_number == 2:
            if datasettype == 'train':
                slide_interval = 5
            elif datasettype == 'val':
                slide_interval = 12
            else:
                slide_interval = 12
        elif fold_number == 3:
            if datasettype == 'train':
                slide_interval = 7
            elif datasettype == 'val':
                slide_interval = 10
            else:
                slide_interval = 12
        elif fold_number == 4:
            if datasettype == 'train':
                slide_interval = 7
            elif datasettype == 'val':
                slide_interval = 15
            else:
                slide_interval = 12
        elif fold_number == 5:
            if datasettype == 'train':
                slide_interval = 2
            elif datasettype == 'val':
                slide_interval = 2
            else:
                slide_interval = 12
    else:
        if fold_number == 2:
            if datasettype == 'train':
                slide_interval = 50
            elif datasettype == 'val':
                slide_interval = 50
            else:
                slide_interval = 50
        elif fold_number == 3:
            if datasettype == 'train':
                slide_interval = 50
            elif datasettype == 'val':
                slide_interval = 50
        elif fold_number == 4:
            if datasettype == 'train':
                slide_interval = 50
            elif datasettype == 'val':
                slide_interval = 50
    files = os.listdir(skeleton_data_folder)
    filenames = [file for file in files if file.startswith(filename_prefix)]
    skeleton_file_path = os.path.join(skeleton_data_folder, filenames[0])
    skeleton_data = pd.read_csv(skeleton_file_path)

    neutral_segments = []
    for start_ts, end_ts in neutral_intervals:
        cropped_data = skeleton_data[(skeleton_data['timestamp'] >= start_ts) & (skeleton_data['timestamp'] < end_ts)]
        if len(cropped_data) > segment_length * 30:  # 假设采样率为30Hz
            for start in range(0, len(cropped_data) - int(segment_length * 30), slide_interval):
                segment = cropped_data.iloc[start:start + int(segment_length * 30),1:67]
                neutral_segments.append(segment)
        elif len(cropped_data) == segment_length * 30:
            neutral_segments.append(cropped_data.iloc[:,1:67])

    return neutral_segments


# 处理文件并生成手势数据
def process_files(file_list, gesture_info, skeleton_data_folder, fold_number, datasettype, gesture_definition):
    all_segments = []

    for filename in file_list:
        filename = filename.split('.', 1)[0]
        skeleton_file_name_begin = 'video' + filename.strip()

        gestures = gesture_info[gesture_info['Filename'].str.startswith(skeleton_file_name_begin, na=False)]

        if gestures.empty:
            print(f"No gesture info found for {filename}, skipping.")
            continue

        skeleton_file = os.path.join(skeleton_data_folder, gestures['Filename'].unique()[0])
        if not os.path.exists(skeleton_file):
            print(f"File {skeleton_file} not found, skipping.")
            continue

        skeleton_data = pd.read_csv(skeleton_file)

        start_ts_list_explicit = []
        end_ts_list_explicit = []
        start_ts_list_implicit = []
        end_ts_list_implicit = []

        # 遍历每个手势并裁剪相应数据段
        for _, row in gestures.iterrows():
            gesture_type = row['Gesture Type']
            start_ts = row['Start Timestamp']
            end_ts = row['End Timestamp']

            if gesture_definition == 'explicit':
                if gesture_type in ['Right_Hand_Explicit', 'Left_Hand_Explicit']:
                    start_ts_list_explicit.append(start_ts)
                    end_ts_list_explicit.append(end_ts)
            elif gesture_definition == 'implicit':
                if gesture_type in ['Left_Look', 'Left_Overshoulder', 'Right_Look']:
                    start_ts_list_implicit.append(start_ts)
                    end_ts_list_implicit.append(end_ts)

            segments, gesture_type = process_skeleton_file(row['Filename'], start_ts, end_ts, gesture_type, skeleton_data_folder, fold_number, datasettype, gesture_definition)
            if gesture_type != 'Skip':
                all_segments.extend([(seg, gesture_type) for seg in segments])
        
        if gesture_definition == 'explicit':
            neutral_intervals_explicit = get_neutral_intervals(start_ts_list_explicit, end_ts_list_explicit)
            neutral_explicit = process_neutral_segments(skeleton_file_name_begin, neutral_intervals_explicit, skeleton_data_folder, fold_number, datasettype, gesture_definition)
            all_segments.extend([(seg, 'neutral_explicit') for seg in neutral_explicit])
        elif gesture_definition == 'implicit':
            neutral_intervals_implicit = get_neutral_intervals(start_ts_list_implicit, end_ts_list_implicit)
            neutral_implicit = process_neutral_segments(skeleton_file_name_begin, neutral_intervals_implicit, skeleton_data_folder, fold_number, datasettype, gesture_definition)
            all_segments.extend([(seg, 'neutral_implicit') for seg in neutral_implicit])

    return all_segments


# 裁剪手势数据
def CropGestureData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number, gesture_definition):
    gesture_info = pd.read_csv(gesture_info_path)

    # # Delete the incomlete lines
    # gesture_info_clean = gesture_info.dropna()

    # 处理训练、验证、测试数据
    train_data = process_files(train_files, gesture_info, skeleton_data_folder, fold_number, 'train', gesture_definition)
    val_data = process_files(val_files, gesture_info, skeleton_data_folder, fold_number, 'val', gesture_definition)
    test_data = process_files(test_files, gesture_info, skeleton_data_folder, fold_number, 'test', gesture_definition)

    # Balance test_data so that each gesture type has the same number of segments as 'Left_Overshoulder'
    if gesture_definition == "explicit":
        train_data = balance_gesture_data(train_data, target_gesture='Right_Hand_Explicit')
        val_data = balance_gesture_data(val_data, target_gesture='Right_Hand_Explicit')
        test_data = balance_gesture_data(test_data, target_gesture='Right_Hand_Explicit')
    elif gesture_definition == "implicit":
        train_data = balance_gesture_data(train_data, target_gesture='Left_Overshoulder')
        val_data = balance_gesture_data(val_data, target_gesture='Left_Overshoulder')
        test_data = balance_gesture_data(test_data, target_gesture='Left_Overshoulder')

   

    all_segments = train_data + val_data + test_data

    with open(os.path.join(output_dir, f'fold_{fold_number}_data_Manuever_Type_Counts.txt'), 'w') as f:
        print(f"Manuever Type Counts fold {fold_number}:")
        f.write("Manuever Type Counts:\n")
        for dataset, name in [(train_data, "Training"), (val_data, "Validation"), (test_data, "Testing")]:
            manuever_counts = Counter([manuever_type for _, manuever_type in dataset])
            print(f"{name} Set:")
            f.write(f"{name}:\n")
            for manuever_type, count in manuever_counts.items():
                print(f"{manuever_type}: {count} segments")
                f.write(f"{manuever_type}: {count} segments\n")

    if fold_number == 5:
        print(f"Total segments: {len(all_segments)}")

        gesture_counts = Counter([gesture_type for _, gesture_type in all_segments])

        # 打印统计结果
        print("Gesture Type Counts:")
        for gesture_type, count in gesture_counts.items():
            print(f"{gesture_type}: {count} segments")

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
explicit_output_dir = 'Gesture2Manuever_Prediction/GestureClassification/ExplicitGestureDataset'
implicit_output_dir = 'Gesture2Manuever_Prediction/GestureClassification/ImplicitGestureDataset'

for fold_number in range(1, 6):
    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f]

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_val_files.txt', 'r') as f:
        val_files = [line.strip() for line in f]

    with open(f'Gesture2Manuever_Prediction/VideoCrop/fold_{fold_number}_test_files.txt', 'r') as f:
        test_files = [line.strip() for line in f]

    CropGestureData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, explicit_output_dir, fold_number, 'explicit')
    CropGestureData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, implicit_output_dir, fold_number, 'implicit')

    # Release memory after each fold
    del train_files, val_files, test_files  # Free memory used by the file lists
    gc.collect()  # Run garbage collection to free up memory
