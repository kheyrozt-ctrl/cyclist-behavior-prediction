#!/usr/bin/env python3
"""
fallback_dnn.py — Lightweight skeleton detection using OpenCV DNN (no PyTorch/TRT needed).

Uses OpenPose COCO Caffe model for CPU-based inference (~2-5 FPS).
Useful for quick validation without the full TensorRT stack.

Usage:
    python3 fallback_dnn.py
    python3 fallback_dnn.py --headless -o out.avi
    python3 fallback_dnn.py --webcam    # Use webcam instead of RealSense
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# COCO keypoints (18 points)
BODY_PARTS = {
    0: "Nose", 1: "Neck",
    2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist",
    8: "RHip", 9: "RKnee", 10: "RAnkle",
    11: "LHip", 12: "LKnee", 13: "LAnkle",
    14: "REye", 15: "LEye", 16: "REar", 17: "LEar",
}

POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# Colors: left=blue, right=red, center=green
PAIR_COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 255),  # right shoulder chain, left shoulder chain start
    (255, 0, 0), (255, 0, 0),                                # left arm
    (0, 0, 255), (0, 0, 255), (0, 0, 255),                  # right leg
    (255, 0, 0), (255, 0, 0), (255, 0, 0),                  # left leg
    (0, 255, 0), (0, 0, 255), (0, 0, 255),                  # nose, right eye/ear
    (255, 0, 0), (255, 0, 0),                                # left eye/ear
]


def get_camera(use_webcam=False):
    """Get camera source: RealSense or webcam."""
    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("ERROR: Cannot open webcam.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap, None

    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        # Warm up
        for _ in range(30):
            pipeline.wait_for_frames()
        return None, pipeline
    except (ImportError, RuntimeError) as e:
        print(f"RealSense not available ({e}), falling back to webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("ERROR: No camera available.")
        return cap, None


def get_frame(cap, pipeline):
    """Get a frame from either webcam or RealSense."""
    if pipeline is not None:
        import pyrealsense2 as rs
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())
    else:
        ret, frame = cap.read()
        return frame if ret else None


def run(args):
    """Main detection loop using OpenCV DNN."""
    proto = os.path.join(MODELS_DIR, "pose_deploy_linevec.prototxt")
    model = os.path.join(MODELS_DIR, "pose_iter_440000.caffemodel")

    if not os.path.exists(proto) or not os.path.exists(model):
        print("ERROR: DNN model files not found.")
        print(f"  Expected: {proto}")
        print(f"  Expected: {model}")
        print("  Run: bash download_models.sh")
        sys.exit(1)

    print("Loading OpenCV DNN model...")
    net = cv2.dnn.readNetFromCaffe(proto, model)
    # Use CUDA if available, else CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA backend for DNN.")
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU backend for DNN (slow, ~2-5 FPS).")

    cap, pipeline = get_camera(args.webcam)

    writer = None
    if args.headless:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))
        print(f"Headless mode: saving to {args.output}")
    else:
        print("Press 'q' to quit.")

    threshold = 0.1
    input_w, input_h = 368, 368

    fps_start = time.time()
    frame_count = 0

    try:
        while True:
            frame = get_frame(cap, pipeline)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Prepare input blob
            blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (input_w, input_h),
                                         (0, 0, 0), swapRB=False, crop=False)
            net.setInput(blob)
            output = net.forward()

            out_h, out_w = output.shape[2], output.shape[3]

            # Extract keypoints
            points = []
            for i in range(len(BODY_PARTS)):
                heatmap = output[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatmap)
                x = int((point[0] / out_w) * w)
                y = int((point[1] / out_h) * h)
                points.append((x, y) if conf > threshold else None)

            # Draw skeleton
            for idx, pair in enumerate(POSE_PAIRS):
                a, b = pair
                if points[a] and points[b]:
                    color = PAIR_COLORS[idx] if idx < len(PAIR_COLORS) else (0, 255, 255)
                    cv2.line(frame, points[a], points[b], color, 2)

            # Draw keypoints
            for i, pt in enumerate(points):
                if pt:
                    # left=blue, right=red, center=green
                    if i in (5, 6, 7, 11, 12, 13, 15, 17):
                        color = (255, 0, 0)
                    elif i in (2, 3, 4, 8, 9, 10, 14, 16):
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.circle(frame, pt, 5, color, -1)

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f} (DNN fallback)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if elapsed > 2.0:
                fps_start = time.time()
                frame_count = 0

            if writer:
                writer.write(frame)
            else:
                cv2.imshow("OpenCV DNN Pose Detection (Fallback)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        if writer:
            writer.release()
            print(f"Video saved to: {args.output}")
        cv2.destroyAllWindows()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Skeleton detection using OpenCV DNN (fallback)")
    parser.add_argument("--headless", action="store_true",
                        help="Save output to video file instead of displaying")
    parser.add_argument("-o", "--output", default="pose_fallback_output.avi",
                        help="Output video path (default: pose_fallback_output.avi)")
    parser.add_argument("--webcam", action="store_true",
                        help="Use webcam instead of RealSense")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
