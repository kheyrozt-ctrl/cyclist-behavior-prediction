import os
import subprocess

# 指定视频文件所在的目录
video_directory = './GestureDataset/depth_camera_videos_croped'

# 遍历目录中的所有文件
for filename in os.listdir(video_directory):
    if filename.endswith('.mp4'):
        video_path = os.path.join(video_directory, filename)
        # 构造命令
        command = f"python .\\Gesture2Manuever_Prediction\\main.py --video {video_path} --record_csv"
        # 执行命令
        subprocess.run(command, shell=True)