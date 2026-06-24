#!/usr/bin/env python3
"""Real-time cyclist intention prediction using RealSense D435 + Pose Estimation + Bi-LSTM.

Pipeline:
  RealSense D435 (640x480@30fps)
  -> Pose Estimation (33 keypoints -> 66 features)
     Backends: MediaPipe BlazePose or NVIDIA trt_pose (via --pose flag)
  -> Sliding window (30 frames)
  -> Explicit + Implicit gesture classifiers (Bi-LSTM)
  -> Gesture weight encoding -> gesture buffer (719 entries)
  -> Maneuver predictor (Bi-LSTM -> Crossing/LeftTurn/RightTurn)
  -> Display overlay
"""

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np
import torch
import pyrealsense2 as rs

from models import (
    ExplicitLSTMClassifier,
    ImplicitLSTMClassifier,
    ManeuverLSTMClassifier,
)
from pose_adapter import create_pose_adapter

# --- Constants ---

# Explicit gesture: OrdinalEncoder order from newsplit_dataloader.py:10
# [0]=Right_Hand_Explicit, [1]=neutral_explicit, [2]=Left_Hand_Explicit
EXPLICIT_LABELS = ["Right_Hand_Explicit", "neutral_explicit", "Left_Hand_Explicit"]
EXPLICIT_WEIGHTS = {
    "Right_Hand_Explicit": 1,
    "neutral_explicit": 0,
    "Left_Hand_Explicit": -1,
}

# Implicit gesture: OrdinalEncoder order from newsplit_dataloader.py:11
# [0]=Right_Look, [1]=neutral_implicit, [2]=Left_Overshoulder, [3]=Left_Look
IMPLICIT_LABELS = ["Right_Look", "neutral_implicit", "Left_Overshoulder", "Left_Look"]
IMPLICIT_WEIGHTS = {
    "Right_Look": 1,
    "neutral_implicit": 0,
    "Left_Overshoulder": -2,
    "Left_Look": -1,
}

# Maneuver classes: LabelEncoder alphabetical order
MANEUVER_LABELS = ["Crossing", "LeftTurn", "RightTurn"]

WINDOW_SIZE = 30       # frames for gesture classification
GESTURE_SEQ_LEN = 719  # frames for maneuver prediction

# Default model paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, os.pardir, "cyclistprediction",
                        "Gesture2Manuever_Prediction")
DEFAULT_EXPLICIT = os.path.join(BASE_DIR, "GestureClassification",
                                "ExplicitModels", "best_model_fold_{}.pkl")
DEFAULT_IMPLICIT = os.path.join(BASE_DIR, "GestureClassification",
                                "ImplicitModels", "best_model_fold_{}.pkl")
DEFAULT_MANEUVER = os.path.join(BASE_DIR, "ManueverPrediction",
                                "Models", "best_model_fold_{}.pkl")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time cyclist intention prediction")
    p.add_argument("--explicit-model", type=str, default=None,
                   help="Path to explicit gesture model weights (.pkl)")
    p.add_argument("--implicit-model", type=str, default=None,
                   help="Path to implicit gesture model weights (.pkl)")
    p.add_argument("--maneuver-model", type=str, default=None,
                   help="Path to maneuver model weights (.pkl)")
    p.add_argument("--fold", type=int, default=5, choices=range(1, 6),
                   help="Model fold to use (1-5, default: 5)")
    p.add_argument("--headless", action="store_true",
                   help="No display window; save to video file instead")
    p.add_argument("-o", "--output", type=str, default="output.avi",
                   help="Output video path (used with --headless)")
    p.add_argument("--pose", type=str, default="mediapipe",
                   choices=["mediapipe", "trt"],
                   help="Pose estimation backend (default: mediapipe)")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (default: cuda if available, else cpu)")
    p.add_argument("--duration", type=float, default=None,
                   help="Optional run duration in seconds, useful for tests")
    return p.parse_args()


def resolve_model_paths(args):
    """Resolve model paths from args or defaults using fold number."""
    fold = args.fold
    explicit = args.explicit_model or DEFAULT_EXPLICIT.format(fold)
    implicit = args.implicit_model or DEFAULT_IMPLICIT.format(fold)
    maneuver = args.maneuver_model or DEFAULT_MANEUVER.format(fold)
    for name, path in [("Explicit", explicit), ("Implicit", implicit),
                       ("Maneuver", maneuver)]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {name} model not found: {path}")
    return explicit, implicit, maneuver


def load_models(explicit_path, implicit_path, maneuver_path, device):
    """Load all three models and set to eval mode."""
    explicit_model = ExplicitLSTMClassifier()
    implicit_model = ImplicitLSTMClassifier()
    maneuver_model = ManeuverLSTMClassifier()

    explicit_model.load_state_dict(torch.load(explicit_path, map_location=device))
    implicit_model.load_state_dict(torch.load(implicit_path, map_location=device))
    maneuver_model.load_state_dict(torch.load(maneuver_path, map_location=device))

    explicit_model.to(device).eval()
    implicit_model.to(device).eval()
    maneuver_model.to(device).eval()

    print(f"Models loaded on {device}")
    for name, m in [("Explicit", explicit_model), ("Implicit", implicit_model),
                    ("Maneuver", maneuver_model)]:
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n_params:,} parameters")

    return explicit_model, implicit_model, maneuver_model


def init_realsense(width=640, height=480, fps=30):
    """Initialize RealSense pipeline for color stream."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    sensor = profile.get_device().first_color_sensor()
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1)

    # Allow auto-exposure to settle
    print("Warming up camera (30 frames)...")
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline


def classify_gesture(model, keypoint_buffer, device):
    """Run gesture classifier on a 30-frame window. Returns class index."""
    window = np.array(keypoint_buffer, dtype=np.float32)  # (30, 66)
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 30, 66)
    with torch.no_grad():
        output = model(tensor)
    return torch.argmax(output, dim=1).item()


def predict_maneuver(model, gesture_buffer, device):
    """Run maneuver predictor on 719-frame gesture sequence. Returns class index."""
    seq = np.array(gesture_buffer, dtype=np.float32)  # (719, 2)
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 719, 2)
    with torch.no_grad():
        output = model(tensor)
    return torch.argmax(output, dim=1).item()


def draw_overlay(frame, explicit_label, implicit_label,
                 maneuver_label, gesture_fill, fps):
    """Draw prediction text overlay on frame."""
    # Text overlay
    y_offset = 30
    lines = [
        f"FPS: {fps:.1f}",
        f"Explicit: {explicit_label or 'waiting...'}",
        f"Implicit: {implicit_label or 'waiting...'}",
        f"Maneuver: {maneuver_label or 'collecting...'}",
        f"Gesture buffer: {gesture_fill}/{GESTURE_SEQ_LEN}",
    ]
    for i, text in enumerate(lines):
        color = (0, 255, 0) if "waiting" not in text and "collecting" not in text \
            else (0, 255, 255)
        # Maneuver line gets special coloring
        if text.startswith("Maneuver:") and maneuver_label:
            if "Left" in maneuver_label:
                color = (255, 100, 0)   # blue-ish for left
            elif "Right" in maneuver_label:
                color = (0, 100, 255)   # red-ish for right
            else:
                color = (0, 255, 0)     # green for crossing
        cv2.putText(frame, text, (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # shadow
        cv2.putText(frame, text, (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Progress bar for gesture buffer fill
    bar_y = y_offset + len(lines) * 30 + 10
    bar_width = 300
    fill_ratio = gesture_fill / GESTURE_SEQ_LEN
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(bar_width * fill_ratio), bar_y + 15),
                  (0, 255, 0), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 15), (255, 255, 255), 1)

    return frame


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load models
    explicit_path, implicit_path, maneuver_path = resolve_model_paths(args)
    explicit_model, implicit_model, maneuver_model = load_models(
        explicit_path, implicit_path, maneuver_path, device)

    # Init camera
    pipeline = init_realsense()

    # Init pose estimation
    pose = create_pose_adapter(args.pose)

    # Buffers
    keypoint_buffer = collections.deque(maxlen=WINDOW_SIZE)
    gesture_buffer = collections.deque(maxlen=GESTURE_SEQ_LEN)

    # Video writer for headless mode
    writer = None
    if args.headless:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))
        print(f"Recording to {args.output}")

    # State
    explicit_label = None
    implicit_label = None
    maneuver_label = None
    frame_count = 0
    fps = 0.0
    t_start = time.time()
    run_start = time.time()

    print("Starting inference loop. Press 'q' to quit.")

    try:
        while True:
            if args.duration is not None and time.time() - run_start >= args.duration:
                print(f"Reached duration limit: {args.duration:.1f}s")
                break

            # Grab frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # Pose estimation
            pose.process(frame)
            keypoints = pose.extract_keypoints()
            if keypoints is not None:
                keypoint_buffer.append(keypoints)

            # Gesture classification when we have a full window
            if len(keypoint_buffer) == WINDOW_SIZE:
                ex_idx = classify_gesture(explicit_model, keypoint_buffer, device)
                im_idx = classify_gesture(implicit_model, keypoint_buffer, device)

                explicit_label = EXPLICIT_LABELS[ex_idx]
                implicit_label = IMPLICIT_LABELS[im_idx]

                ex_weight = EXPLICIT_WEIGHTS[explicit_label]
                im_weight = IMPLICIT_WEIGHTS[implicit_label]
                gesture_buffer.append([ex_weight, im_weight])

            # Maneuver prediction when gesture buffer is full
            if len(gesture_buffer) == GESTURE_SEQ_LEN:
                man_idx = predict_maneuver(maneuver_model, gesture_buffer, device)
                maneuver_label = MANEUVER_LABELS[man_idx]

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Draw overlay
            pose.draw_skeleton(frame)
            frame = draw_overlay(frame, explicit_label, implicit_label,
                                 maneuver_label, len(gesture_buffer), fps)

            if args.headless:
                writer.write(frame)
            else:
                cv2.imshow("Cyclist Intention Prediction", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pose.close()
        pipeline.stop()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
