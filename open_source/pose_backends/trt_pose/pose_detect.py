#!/usr/bin/env python3
"""
pose_detect.py — Real-time skeleton pose detection using trt_pose + RealSense D435.

Usage:
    python3 pose_detect.py              # Live display
    python3 pose_detect.py --headless   # Save to output video (for SSH)
    python3 pose_detect.py --headless -o out.avi  # Custom output path
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category (inlined to avoid pycocotools dep)."""
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology

# COCO 17 keypoint colors: left side blue, right side red, center green
KEYPOINT_COLORS = {
    0: (0, 255, 0),     # nose - green
    1: (0, 255, 0),     # left_eye - green
    2: (0, 255, 0),     # right_eye - green
    3: (255, 0, 0),     # left_ear - blue
    4: (0, 0, 255),     # right_ear - red
    5: (255, 0, 0),     # left_shoulder - blue
    6: (0, 0, 255),     # right_shoulder - red
    7: (255, 0, 0),     # left_elbow - blue
    8: (0, 0, 255),     # right_elbow - red
    9: (255, 0, 0),     # left_wrist - blue
    10: (0, 0, 255),    # right_wrist - red
    11: (255, 0, 0),    # left_hip - blue
    12: (0, 0, 255),    # right_hip - red
    13: (255, 0, 0),    # left_knee - blue
    14: (0, 0, 255),    # right_knee - red
    15: (255, 0, 0),    # left_ankle - blue
    16: (0, 0, 255),    # right_ankle - red
}


def check_realsense_camera():
    """Check if a RealSense camera is connected."""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("ERROR: No RealSense camera detected.")
            print("  - Check USB connection (use USB 3.0 port)")
            print("  - Run: lsusb | grep 8086")
            print("  - Expected: Intel Corp. (8086:0b5c for D435)")
            return False
        for dev in devices:
            print(f"Found: {dev.get_info(rs.camera_info.name)} "
                  f"(S/N: {dev.get_info(rs.camera_info.serial_number)})")
        return True
    except ImportError:
        print("ERROR: pyrealsense2 not installed. Run: bash setup.sh")
        return False


def init_realsense(width=640, height=480, fps=30):
    """Initialize RealSense pipeline for color stream."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    sensor = profile.get_device().first_color_sensor()
    # Enable auto-exposure
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1)

    # Allow auto-exposure to settle
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline


def load_trt_pose_model():
    """Load trt_pose model, with TensorRT optimization if available."""
    from trt_pose.parse_objects import ParseObjects
    import trt_pose.models

    # Load topology
    topology_path = os.path.join(MODELS_DIR, "human_pose.json")
    if not os.path.exists(topology_path):
        sys.exit(f"ERROR: Topology file not found: {topology_path}\nRun: bash download_models.sh")

    with open(topology_path, "r") as f:
        human_pose = json.load(f)

    topology = coco_category_to_topology(human_pose)
    num_parts = len(human_pose["keypoints"])
    num_links = len(human_pose["skeleton"])

    # Build model
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

    # Load weights
    weights_path = os.path.join(MODELS_DIR, "resnet18_baseline_att_224x224_A_epoch_249.pth")
    if not os.path.exists(weights_path):
        sys.exit(f"ERROR: Model weights not found: {weights_path}\nRun: bash download_models.sh")

    model.load_state_dict(torch.load(weights_path))

    # Try TensorRT optimization, fall back to PyTorch CUDA if it fails
    trt_cache = os.path.join(MODELS_DIR, "resnet18_baseline_att_224x224_A_trt.pth")
    use_trt = False

    if os.path.exists(trt_cache):
        try:
            from torch2trt import TRTModule
            print("Loading cached TensorRT model...")
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_cache))
            model = model_trt
            use_trt = True
        except Exception as e:
            print(f"Failed to load TRT cache ({e}), using PyTorch CUDA.")
    else:
        try:
            from torch2trt import TRTModule, torch2trt
            print("Optimizing model with TensorRT (first run only, ~30s)...")
            data = torch.zeros((1, 3, 224, 224)).cuda()
            model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25)
            torch.save(model_trt.state_dict(), trt_cache)
            print(f"TensorRT model cached to: {trt_cache}")
            model = model_trt
            use_trt = True
        except Exception as e:
            print(f"TensorRT optimization failed ({e}).")
            print("Using PyTorch CUDA inference instead (still fast on Orin).")

    if use_trt:
        print("Backend: TensorRT")
    else:
        print("Backend: PyTorch CUDA")

    parse_objects = ParseObjects(topology)

    return model, parse_objects, topology, human_pose


def preprocess_frame(frame):
    """Preprocess frame for trt_pose: resize, normalize, to CUDA tensor."""

    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img


def draw_skeleton(frame, counts, objects, peaks, topology, human_pose):
    """Draw keypoints and skeleton on the frame."""
    h, w = frame.shape[:2]
    keypoint_names = human_pose["keypoints"]
    skeleton = human_pose["skeleton"]

    for i in range(counts[0]):
        obj = objects[0][i]
        keypoints = {}

        # Draw keypoints
        for j in range(len(keypoint_names)):
            k = int(obj[j])
            if k >= 0:
                peak = peaks[0][j][k]
                x = int(peak[1] * w)
                y = int(peak[0] * h)
                keypoints[j] = (x, y)
                color = KEYPOINT_COLORS.get(j, (0, 255, 255))
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 7, color, 1)

        # Draw skeleton lines
        for link in skeleton:
            a, b = link[0] - 1, link[1] - 1  # COCO skeleton is 1-indexed
            if a in keypoints and b in keypoints:
                # Color based on side
                if a in (3, 5, 7, 9, 11, 13, 15) or b in (3, 5, 7, 9, 11, 13, 15):
                    color = (255, 0, 0)  # left - blue
                elif a in (4, 6, 8, 10, 12, 14, 16) or b in (4, 6, 8, 10, 12, 14, 16):
                    color = (0, 0, 255)  # right - red
                else:
                    color = (0, 255, 0)  # center - green
                cv2.line(frame, keypoints[a], keypoints[b], color, 2)

    return frame


def run(args):
    """Main detection loop."""

    if not check_realsense_camera():
        sys.exit(1)

    print("Initializing RealSense camera...")
    pipeline = init_realsense()

    print("Loading trt_pose model...")
    model_trt, parse_objects, topology, human_pose = load_trt_pose_model()

    writer = None
    if args.headless:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, 30.0, (640, 480))
        print(f"Headless mode: saving to {args.output}")
    else:
        print("Press 'q' to quit.")

    fps_start = time.time()
    frame_count = 0

    try:
        while True:
            import pyrealsense2 as rs
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Inference
            tensor = preprocess_frame(frame)
            with torch.no_grad():
                cmap, paf = model_trt(tensor)
            cmap = cmap.detach().cpu()
            paf = paf.detach().cpu()

            counts, objects, peaks = parse_objects(cmap, paf)

            # Draw
            frame = draw_skeleton(frame, counts, objects, peaks, topology, human_pose)

            # FPS overlay
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Reset FPS counter every 2 seconds
            if elapsed > 2.0:
                fps_start = time.time()
                frame_count = 0

            if writer:
                writer.write(frame)
            else:
                cv2.imshow("trt_pose Skeleton Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.stop()
        if writer:
            writer.release()
            print(f"Video saved to: {args.output}")
        cv2.destroyAllWindows()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Real-time skeleton pose detection with trt_pose + RealSense D435")
    parser.add_argument("--headless", action="store_true",
                        help="Save output to video file instead of displaying (for SSH)")
    parser.add_argument("-o", "--output", default="pose_output.avi",
                        help="Output video path when using --headless (default: pose_output.avi)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
