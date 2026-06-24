#!/usr/bin/env python3
"""Real-time bus-stop cyclist behavior prediction.

Pipeline:
  Camera frame -> 33 pose keypoints -> Stage1 head/upper/leg TorchScript models
  -> 10 seconds of concatenated Stage1 features -> Stage2 maneuver model.
"""

import argparse
import collections
import json
import os
import sys
import time

import cv2
import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
CYCLIST_INFERENCE_DIR = os.path.join(WORKSPACE_DIR, "cyclist_inference")
if CYCLIST_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, CYCLIST_INFERENCE_DIR)

from pose_adapter import create_pose_adapter  # noqa: E402


DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "model_config.json")
DEFAULT_HEAD = os.path.join(SCRIPT_DIR, "head_model.pt")
DEFAULT_UPPER = os.path.join(SCRIPT_DIR, "upper_model.pt")
DEFAULT_LEG = os.path.join(SCRIPT_DIR, "leg_model.pt")
DEFAULT_STAGE2 = os.path.join(SCRIPT_DIR, "stage2_maneuver_model.pt")

HEAD_WINDOW = 12
UPPER_WINDOW = 12
LEG_WINDOW = 20
STAGE2_SEQ_LEN = 120
STAGE1_FPS = 12.0
LEG_FPS = 20.0


class RateSampler:
    """Keep an approximate fixed-rate stream from an arbitrary camera FPS."""

    def __init__(self, fps):
        self.interval = 1.0 / fps
        self.last_sample_time = None

    def should_sample(self, now):
        if self.last_sample_time is None:
            self.last_sample_time = now
            return True
        if now - self.last_sample_time >= self.interval:
            self.last_sample_time = now
            return True
        return False


class RealSenseSource:
    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipeline.start(config)
        sensor = profile.get_device().first_color_sensor()
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        print("Warming up RealSense camera...")
        for _ in range(30):
            self._pipeline.wait_for_frames()

    def read(self):
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self):
        self._pipeline.stop()


class WebcamSource:
    def __init__(self, index=0, width=640, height=480, fps=30):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open webcam index {index}")

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time bus-stop cyclist behavior prediction"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to model_config.json")
    parser.add_argument("--head-model", default=DEFAULT_HEAD,
                        help="Path to Stage1 head TorchScript model")
    parser.add_argument("--upper-model", default=DEFAULT_UPPER,
                        help="Path to Stage1 upper-limb TorchScript model")
    parser.add_argument("--leg-model", default=DEFAULT_LEG,
                        help="Path to Stage1 leg TorchScript model")
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE2,
                        help="Path to Stage2 maneuver TorchScript model")
    parser.add_argument("--pose", default="mediapipe", choices=["mediapipe", "trt"],
                        help="Pose estimation backend")
    parser.add_argument("--camera", default="realsense",
                        choices=["realsense", "webcam"],
                        help="Camera source")
    parser.add_argument("--webcam-index", type=int, default=0,
                        help="OpenCV webcam index when --camera webcam is used")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera capture width")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera capture height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Camera capture FPS")
    parser.add_argument("--device", default=None,
                        help="Reserved for re-exported models; current TorchScript package runs on CPU")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a display window; write annotated video")
    parser.add_argument("-o", "--output", default="bus_stop_output.avi",
                        help="Output video path for headless mode")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional run duration in seconds, useful for tests")
    parser.add_argument("--swap-upper-labels", action="store_true",
                        help="Swap left/right upper-limb display labels if the exported model order is reversed")
    return parser.parse_args()


def require_file(path, label):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {label} not found: {path}")


def load_config(path):
    require_file(path, "Config")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_torchscript(path, device, label):
    require_file(path, label)
    model = torch.jit.load(path, map_location=device)
    model.to(device).eval()
    return model


def load_models(args, device):
    models = {
        "head": load_torchscript(args.head_model, device, "Head model"),
        "upper": load_torchscript(args.upper_model, device, "Upper-limb model"),
        "leg": load_torchscript(args.leg_model, device, "Leg model"),
        "stage2": load_torchscript(args.stage2_model, device, "Stage2 model"),
    }
    print(f"Loaded TorchScript models on {device}")
    return models


def init_camera(args):
    if args.camera == "realsense":
        return RealSenseSource(args.width, args.height, args.fps)
    return WebcamSource(args.webcam_index, args.width, args.height, args.fps)


def softmax_label(logits, labels):
    probs = torch.softmax(logits, dim=1)
    idx = int(torch.argmax(probs, dim=1).item())
    return labels[idx], float(probs[0, idx].item()), idx


def run_stage1(models, buffers, labels, device):
    head_x = torch.tensor(np.array(buffers["head"], dtype=np.float32),
                          device=device).unsqueeze(0)
    upper_x = torch.tensor(np.array(buffers["upper"], dtype=np.float32),
                           device=device).unsqueeze(0)
    leg_x = torch.tensor(np.array(buffers["leg"], dtype=np.float32),
                         device=device).unsqueeze(0)

    with torch.no_grad():
        head_logits, head_feature = models["head"](head_x)
        upper_logits, upper_feature = models["upper"](upper_x)
        leg_logits, leg_feature = models["leg"](leg_x)

    feature = torch.cat([head_feature, upper_feature, leg_feature], dim=1)
    if feature.shape[1] != 640:
        raise RuntimeError(f"Stage1 feature dim should be 640, got {feature.shape[1]}")

    predictions = {
        "head": softmax_label(head_logits, labels["head"]),
        "upper": softmax_label(upper_logits, labels["upper_limb"]),
        "leg": softmax_label(leg_logits, labels["leg"]),
    }
    return feature.squeeze(0).detach().cpu().numpy(), predictions


def run_stage2(model, feature_buffer, labels, device):
    seq = torch.tensor(np.array(feature_buffer, dtype=np.float32),
                       device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(seq)
    return softmax_label(logits, labels)


def draw_text(frame, text, pos, color, scale=0.65):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def draw_overlay(frame, fps, stage1_pred, stage2_pred, fill, pose_ok):
    status = "pose ok" if pose_ok else "no pose"
    lines = [
        (f"FPS: {fps:.1f}  {status}", (0, 255, 0) if pose_ok else (0, 255, 255)),
        (format_prediction("Head", stage1_pred.get("head")), (210, 255, 210)),
        (format_prediction("Upper", stage1_pred.get("upper")), (210, 255, 210)),
        (format_prediction("Leg", stage1_pred.get("leg")), (210, 255, 210)),
        (format_prediction("Maneuver", stage2_pred), (80, 220, 255)),
        (f"Stage2 feature buffer: {fill}/{STAGE2_SEQ_LEN}", (255, 255, 255)),
    ]

    for i, (text, color) in enumerate(lines):
        draw_text(frame, text, (10, 30 + i * 28), color)

    bar_y = 30 + len(lines) * 28 + 8
    bar_width = 320
    ratio = min(1.0, fill / STAGE2_SEQ_LEN)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 14), (45, 45, 45), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(bar_width * ratio), bar_y + 14),
                  (0, 210, 255), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 14), (255, 255, 255), 1)
    return frame


def format_prediction(prefix, pred):
    if pred is None:
        return f"{prefix}: collecting..."
    label, prob, _ = pred
    return f"{prefix}: {label} ({prob:.2f})"


def main():
    args = parse_args()
    config = load_config(args.config)

    labels = {
        "head": config["stage1"]["head"]["classes"],
        "upper_limb": config["stage1"]["upper_limb"]["classes"],
        "leg": config["stage1"]["leg"]["classes"],
        "stage2": config["stage2"]["classes"],
    }
    if args.swap_upper_labels:
        labels["upper_limb"] = labels["upper_limb"].copy()
        labels["upper_limb"][0], labels["upper_limb"][1] = (
            labels["upper_limb"][1],
            labels["upper_limb"][0],
        )
        print("Swapped upper-limb left/right display labels.")

    if args.device and args.device != "cpu":
        print(
            "WARNING: This TorchScript package creates LSTM hidden states on CPU. "
            "Forcing model inference to CPU; re-export the models with device-aware "
            "hidden states before using CUDA."
        )
    device = torch.device("cpu")

    models = load_models(args, device)
    camera = init_camera(args)
    pose = create_pose_adapter(args.pose)

    buffers = {
        "head": collections.deque(maxlen=HEAD_WINDOW),
        "upper": collections.deque(maxlen=UPPER_WINDOW),
        "leg": collections.deque(maxlen=LEG_WINDOW),
    }
    feature_buffer = collections.deque(maxlen=STAGE2_SEQ_LEN)
    stage1_sampler = RateSampler(STAGE1_FPS)
    leg_sampler = RateSampler(LEG_FPS)

    stage1_pred = {}
    stage2_pred = None
    frame_count = 0
    fps = 0.0
    fps_start = time.time()
    run_start = time.time()

    writer = None
    if args.headless:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, float(args.fps),
                                 (args.width, args.height))
        print(f"Headless mode: recording to {args.output}")
    else:
        print("Starting live prediction. Press 'q' to quit.")

    try:
        while True:
            if args.duration is not None and time.time() - run_start >= args.duration:
                print(f"Reached duration limit: {args.duration:.1f}s")
                break

            frame = camera.read()
            if frame is None:
                continue

            now = time.time()
            pose.process(frame)
            keypoints = pose.extract_keypoints()
            pose_ok = keypoints is not None

            if keypoints is not None:
                if leg_sampler.should_sample(now):
                    buffers["leg"].append(keypoints)
                if stage1_sampler.should_sample(now):
                    buffers["head"].append(keypoints)
                    buffers["upper"].append(keypoints)

                    ready = (
                        len(buffers["head"]) == HEAD_WINDOW
                        and len(buffers["upper"]) == UPPER_WINDOW
                        and len(buffers["leg"]) == LEG_WINDOW
                    )
                    if ready:
                        feature, stage1_pred = run_stage1(models, buffers, labels, device)
                        feature_buffer.append(feature)

                        if len(feature_buffer) == STAGE2_SEQ_LEN:
                            stage2_pred = run_stage2(
                                models["stage2"], feature_buffer, labels["stage2"], device
                            )

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            pose.draw_skeleton(frame)
            draw_overlay(frame, fps, stage1_pred, stage2_pred,
                         len(feature_buffer), pose_ok)

            if writer:
                writer.write(frame)
            else:
                cv2.imshow("Bus Stop Cyclist Prediction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pose.close()
        camera.close()
        if writer:
            writer.release()
            print(f"Video saved to: {args.output}")
        if not args.headless:
            cv2.destroyAllWindows()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
