import pandas as pd
import os
import numpy as np
from collections import Counter
import pickle




def process_skeleton_file(filename, start_ts, end_ts, gesture_type, skeleton_data_folder):
    """
    ...
    :param 
    :param 
    :param 
    :return: 
    """
    # 读取对应的skeleton CSV文件
    skeleton_file_path = os.path.join(skeleton_data_folder, filename)
    skeleton_data = pd.read_csv(skeleton_file_path)
    
    # 根据Start Timestamp和End Timestamp进行分割
    cropped_data = skeleton_data[(skeleton_data['timestamp'] >= start_ts) & (skeleton_data['timestamp'] < end_ts)]
    
    # 滑动窗口采样(second)
    segment_length = 1.0  # 每个数据段的长度为1.0秒
    # 动态设置滑动窗口间隔(timestamp)
    if gesture_type == 'Left_Overshoulder':
        slide_interval = 1
    elif gesture_type == 'Left_Look':
        slide_interval = 2
    elif gesture_type == 'Right_Hand_Explicit':
        slide_interval = 4
    elif gesture_type == 'Right_Look':
        slide_interval = 6
    elif gesture_type == 'Left_Hand_Explicit':
        slide_interval = 12
    else:
        slide_interval = 3  # 默认值
    
    segments = []
    if len(cropped_data) > segment_length * 30:  # 假设数据采样率为30Hz
        for start in range(0, len(cropped_data) - int(segment_length * 30), int(slide_interval)):
            segment = cropped_data.iloc[start:start + int(segment_length * 30)]
            segments.append(segment)
    else:
        segments.append(cropped_data)
    
    return segments, gesture_type

def get_neutral_intervals(start_ts_list, end_ts_list, total_duration=24.0):
    """
    获取中立手势的时间段，排除标注手势的前后1.0秒
    :param start_ts_list: 所有标注手势的开始时间戳
    :param end_ts_list: 所有标注手势的结束时间戳
    :param total_duration: 目标时间段的总长度（默认为1.0到28.0）
    :return: 中立手势的时间段列表 [(start, end), ...]
    """
    neutral_intervals = []
    current_start = 1.0
    
    for start_ts, end_ts in sorted(zip(start_ts_list, end_ts_list)):
        neutral_end = start_ts - 1.0
        if neutral_end > current_start:
            neutral_intervals.append((current_start, min(neutral_end, total_duration)))
        current_start = end_ts + 1.0

    # 如果最后一个标注手势的结束时间小于28.0秒，还可以有一段中立手势时间段
    if current_start < total_duration - 1.0:
        neutral_intervals.append((current_start, total_duration))
    
    return neutral_intervals

def process_neutral_segments(filename, neutral_intervals, skeleton_data_folder, segment_length=1.0, slide_interval=1.6):
    """
    ...
    :param 
    :param 
    :param 
    :return: 
    """
    # 读取对应的skeleton CSV文件
    skeleton_file_path = os.path.join(skeleton_data_folder, filename)
    skeleton_data = pd.read_csv(skeleton_file_path)
    
    neutral_segments = []
    
    for start_ts, end_ts in neutral_intervals:
        cropped_data = skeleton_data[(skeleton_data['timestamp'] >= start_ts) & (skeleton_data['timestamp'] < end_ts)]
        
        if len(cropped_data) > segment_length * 30:  # 假设数据采样率为30Hz
            for start in range(0, len(cropped_data) - int(segment_length * 30), int(slide_interval * 30)):
                segment = cropped_data.iloc[start:start + int(segment_length * 30)]
                neutral_segments.append(segment)
        else:
            neutral_segments.append(cropped_data)
    
    return neutral_segments

# 定义函数，用于排除手势前后 1.0 秒的时间段
def exclude_times(times, start_limit, end_limit):
    exclusion_ranges = []
    for start, end in times:
        exclusion_ranges.append((max(start_limit, start - 1.0), min(end_limit, end + 1.0)))
    return exclusion_ranges

def CropGestureData(gesture_info_path,skeleton_data_folder,output_file):
    """
    ...
    :param 
    :param 
    :param 
    :return: 
    """
    gesture_info = pd.read_csv(gesture_info_path)

    # Delete the incomlete lines
    gesture_info_clean = gesture_info.dropna()


    # Traversal gesture_crop_info.csv and process the data 
    all_segments = []
    for index, row in gesture_info_clean.iterrows():
        filename = row['Filename']
        start_ts = row['Start Timestamp']
        end_ts = row['End Timestamp']
        gesture_type = row['Gesture Type']
        
        segments, gesture_type = process_skeleton_file(filename, start_ts, end_ts, gesture_type, skeleton_data_folder)
        all_segments.extend([(seg, gesture_type) for seg in segments])

    # Define both type of neutral segments 
    neutral_explicit_segments = []
    neutral_implicit_segments = []

    for filename in gesture_info_clean['Filename'].unique():
        gestures = gesture_info_clean[gesture_info_clean['Filename'] == filename]
    
        # 定义时间戳列表
        start_ts_list_explicit = []
        end_ts_list_explicit = []
        start_ts_list_implicit = []
        end_ts_list_implicit = []
        
        # 根据 gesture type 分类时间戳
        for _, row in gestures.iterrows():
            gesture_type = row['Gesture Type']
            start_ts = row['Start Timestamp']
            end_ts = row['End Timestamp']
            
            if gesture_type in ['Right_Hand_Explicit', 'Left_Hand_Explicit']:
                start_ts_list_explicit.append(start_ts)
                end_ts_list_explicit.append(end_ts)
            elif gesture_type in ['Left_Look', 'Left_Overshoulder', 'Right_Look']:
                start_ts_list_implicit.append(start_ts)
                end_ts_list_implicit.append(end_ts)
        
        
        neutral_intervals_explicit = get_neutral_intervals(start_ts_list_explicit, end_ts_list_explicit)
        neutral_explicit = process_neutral_segments(filename, neutral_intervals_explicit, skeleton_data_folder)
        neutral_explicit_segments.extend([(seg, 'neutral_explicit') for seg in neutral_explicit])


        neutral_intervals_implicit = get_neutral_intervals(start_ts_list_implicit, end_ts_list_implicit)
        neutral_implicit = process_neutral_segments(filename, neutral_intervals_implicit, skeleton_data_folder)
        neutral_implicit_segments.extend([(seg, 'neutral_implicit') for seg in neutral_implicit])

    neutral_segments_all = neutral_explicit_segments + neutral_implicit_segments

    print(f"Total neutral explicit segments: {len(neutral_explicit_segments)}")
    print(f"Total neutral implicit segments: {len(neutral_implicit_segments)}")
    print(f"Total neutral segments: {len(neutral_segments_all)}")

    all_segments.extend(neutral_segments_all)
    print(f"Total segments: {len(all_segments)}")

    # 统计每种gesture_type的数量
    gesture_counts = Counter([gesture_type for _, gesture_type in all_segments])

    # 打印统计结果
    print("Gesture Type Counts:")
    for gesture_type, count in gesture_counts.items():
        print(f"{gesture_type}: {count} segments")

    
    
    

    # 保存 all_segments 到文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_segments, f)

    print(f"All segments saved to {output_file}")



