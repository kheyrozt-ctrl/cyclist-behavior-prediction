import os
import csv
from datetime import datetime
from mediapipe import solutions as mp
import cv2

def create_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def initialize_csv_file(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['timestamp']
        for landmark in mp.pose.PoseLandmark:
            header.extend([f"{landmark.name}_x", f"{landmark.name}_y"])
        writer.writerow(header)

def log_keypoints_to_csv(landmarks, timestamp, csv_file):
    row = [timestamp]
    for landmark in landmarks:
        # row.extend([landmark.x, landmark.y, landmark.z])
        row.extend([landmark.x, landmark.y])
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def initialize_video_writer(video_file, frame_width, frame_height, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
    return video_writer

def write_frame_to_video(video_writer, frame):
    video_writer.write(frame)

def release_video_writer(video_writer):
    video_writer.release()

def prepare_output(output_dir):
    create_output_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'{output_dir}/{timestamp}_skeleton_keypoints.csv'
    video_file = f'{output_dir}/{timestamp}_output_video.avi'
    initialize_csv_file(csv_file)
    return csv_file, video_file, timestamp

def prepare_output_files(output_dir, frame_width, frame_height, fps=20):
    csv_file, video_file, timestamp = prepare_output(output_dir)
    video_writer = initialize_video_writer(video_file, frame_width, frame_height, fps)
    return csv_file, video_writer
