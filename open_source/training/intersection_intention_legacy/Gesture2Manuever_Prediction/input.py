import cv2
import os

def initialize_camera(cam=0):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video camera {cam}.")
    print(f"Camera {cam} initialized successfully.")
    return cap

def initialize_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Error: The video file at {video_path} does not exist.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file at {video_path}.")
    print(f"Video file {video_path} initialized successfully.")
    return cap
