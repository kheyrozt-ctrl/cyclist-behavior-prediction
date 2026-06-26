import os
import pandas as pd

def generate_crop_info_csv(directory, output_csv, num_crops=5):
    # 获取目录下所有的csv文件
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    # files = ["Full33SkeletonData/example_skeleton.csv"]
    
    # 创建一个空的字典列表
    rows = []
    
    # 对每个文件预留num_crops行
    for file in files:
        for i in range(1, num_crops + 1):
            rows.append({
                'Filename': file,
                'Crop Number': f'Crop {i}',
                'Start Timestamp': '',
                'End Timestamp': ''
            })
    
    # 使用DataFrame从字典列表创建数据
    crop_info = pd.DataFrame.from_records(rows)
    
    # 将DataFrame保存为CSV文件
    crop_info.to_csv(output_csv, index=False)
    print(f"Crop info CSV generated: {output_csv}")

# 使用示例
directory = 'Full33SkeletonData'
output_csv = 'Gesture2Manuever_Prediction/VideoCrop/gesture_crop_info.csv'
generate_crop_info_csv(directory, output_csv)
