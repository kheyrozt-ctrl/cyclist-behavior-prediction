import numpy as np
import mediapipe as mp

def detect_movement(landmarks, mp_pose):
    if not landmarks:
        return "Stop"
    
    # Extract relevant keypoints
    right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    right_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    left_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    
    # Define thresholds
    arm_raise_threshold = 0.1
    leg_movement_threshold = 0.05
    head_turn_threshold = 0.15

    # Check arm raise for right turn
    if (right_elbow.y < right_shoulder.y - arm_raise_threshold or 
        right_wrist.y < right_shoulder.y - arm_raise_threshold):
        return "Right Turn"
    
    # Check head turn for right turn
    if nose.x < right_shoulder.x - head_turn_threshold:
        return "Right Turn"
    
    # Check arm raise for left turn
    if (left_elbow.y < left_shoulder.y - arm_raise_threshold or 
        left_wrist.y < left_shoulder.y - arm_raise_threshold):
        return "Left Turn"
    
    # Check head turn for left turn
    if nose.x > left_shoulder.x + head_turn_threshold:
        return "Left Turn"
    
    # Check if legs are not moving for stop
    if abs(right_knee.y - right_hip.y) < leg_movement_threshold and abs(left_knee.y - left_hip.y) < leg_movement_threshold:
        return "Stop"
    
    # Default movement
    return "Continue"
