#!/usr/bin/env python3
"""Holoscan graph for camera -> cyclist prediction -> display/publication."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
INFERENCE_DIR = os.path.join(ROOT, "open_source", "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from edge_deployment.holoscan.simulation import (
    synthetic_frame,
    synthetic_keypoints,
    synthetic_label,
)

if sys.platform == "win32":
    raise SystemExit(
        "NVIDIA Holoscan SDK is not available as a native Windows Python "
        "runtime. Run this graph on a supported Linux x86_64/CUDA or Jetson "
        "environment. On Windows, use: "
        r".venv\Scripts\python.exe unified_prediction\run_web.py"
    )

try:
    from holoscan.conditions import CountCondition
    from holoscan.core import Application, Operator, OperatorSpec
except ModuleNotFoundError as exc:
    if exc.name != "holoscan":
        raise
    raise SystemExit(
        "Holoscan SDK is not installed in this interpreter. On supported "
        "Linux/Jetson systems run: "
        "python3 -m pip install -r edge_deployment/holoscan/requirements.txt"
    ) from exc


class CameraOp(Operator):
    def __init__(
        self,
        fragment,
        *,
        index=0,
        width=640,
        height=480,
        fps=30,
        duration=0.0,
        **kwargs,
    ):
        super().__init__(fragment, **kwargs)
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.duration = duration
        self.capture = None
        self.started_at = None

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.started_at = time.monotonic()
        self.capture = cv2.VideoCapture(self.index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {self.index}")

    def compute(self, op_input, op_output, context):
        if self.duration > 0 and time.monotonic() - self.started_at >= self.duration:
            raise KeyboardInterrupt
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera frame acquisition failed")
        op_output.emit({"frame": frame, "timestamp": time.time()}, "frame")

    def stop(self):
        if self.capture is not None:
            self.capture.release()


class SyntheticCameraOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        width=640,
        height=480,
        fps=30,
        duration=10.0,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self.width, self.height, self.fps = width, height, fps
        self.duration = duration
        self.started_at = None
        self.started_wall = None
        self.frame_index = 0

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.started_at = time.monotonic()
        self.started_wall = time.time()

    def compute(self, op_input, op_output, context):
        target_time = self.started_at + self.frame_index / max(self.fps, 1)
        delay = target_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        frame = synthetic_frame(
            self.frame_index,
            self.width,
            self.height,
            self.fps,
        )
        timestamp = self.started_wall + self.frame_index / max(self.fps, 1)
        self.frame_index += 1
        op_output.emit({"frame": frame, "timestamp": timestamp}, "frame")


class SyntheticPoseAdapter:
    def __init__(self, fps):
        self.fps = fps
        self.frame_index = 0
        self.keypoints = None

    def process(self, _frame):
        self.keypoints = synthetic_keypoints(self.frame_index, self.fps)
        self.frame_index += 1

    def extract_keypoints(self):
        return self.keypoints

    def draw_skeleton(self, frame):
        if self.keypoints is None:
            return
        height, width = frame.shape[:2]
        for index in range(0, len(self.keypoints), 2):
            x = int(self.keypoints[index] * width)
            y = int(self.keypoints[index + 1] * height)
            cv2.circle(frame, (x, y), 3, (62, 183, 219), -1)

    def close(self):
        pass


class SyntheticPredictor:
    title = "Synthetic graph-validation predictor"

    def __init__(self, fps):
        self.fps = max(int(fps), 1)
        self.frame_count = 0
        self.label = "warming_up"

    def update(self, _keypoints, _timestamp):
        self.frame_count += 1
        self.label = synthetic_label(self.frame_count, self.fps)

    def overlay_lines(self):
        return [
            (f"Synthetic maneuver: {self.label}", (80, 220, 255)),
            (f"Simulation frames: {self.frame_count}", (255, 255, 255)),
        ]

    def progress(self):
        return min(self.frame_count, self.fps), self.fps


class CyclistPredictionOp(Operator):
    def __init__(self, fragment, *, args, **kwargs):
        super().__init__(fragment, **kwargs)
        self.runtime_args = args
        self.pose = self.predictor = None

    def setup(self, spec: OperatorSpec):
        spec.input("frame")
        spec.output("annotated")
        spec.output("prediction")

    def start(self):
        if self.runtime_args.pose == "synthetic":
            self.pose = SyntheticPoseAdapter(self.runtime_args.fps)
        else:
            from intersection_runtime.pose_adapter import create_pose_adapter
            self.pose = create_pose_adapter(self.runtime_args.pose)
        if self.runtime_args.model == "synthetic":
            self.predictor = SyntheticPredictor(self.runtime_args.fps)
        else:
            import torch
            torch.set_num_threads(self.runtime_args.torch_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            from unified_prediction.predictors import create_predictor
            self.predictor = create_predictor(self.runtime_args)
        self.frame_index = 0

    def compute(self, op_input, op_output, context):
        from unified_prediction.runtime import draw_overlay
        packet = op_input.receive("frame")
        if isinstance(packet, dict):
            frame = packet["frame"]
            now = float(packet["timestamp"])
        else:
            frame = packet
            now = time.time()
        self.pose.process(frame)
        keypoints = self.pose.extract_keypoints()
        if keypoints is not None:
            self.predictor.update(keypoints, now)
        self.pose.draw_skeleton(frame)
        lines = self.predictor.overlay_lines()
        draw_overlay(frame, self.predictor.title, 0.0, keypoints is not None,
                     lines, self.predictor.progress())
        op_output.emit(frame, "annotated")
        op_output.emit(
            {
                "frame_index": self.frame_index,
                "timestamp": now,
                "pose_ok": keypoints is not None,
                "progress": self.predictor.progress(),
                "lines": [text for text, _ in lines],
            },
            "prediction",
        )
        self.frame_index += 1

    def stop(self):
        if self.pose is not None:
            self.pose.close()


class OutputOp(Operator):
    def __init__(self, fragment, *, headless=False, output_jsonl=None, **kwargs):
        super().__init__(fragment, **kwargs)
        self.headless = headless
        self.output_jsonl = Path(output_jsonl) if output_jsonl else None
        self.output_handle = None

    def setup(self, spec: OperatorSpec):
        spec.input("frame")
        spec.input("prediction")

    def start(self):
        if self.output_jsonl is not None:
            self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self.output_handle = self.output_jsonl.open("w", encoding="utf-8")

    def compute(self, op_input, op_output, context):
        frame = op_input.receive("frame")
        prediction = op_input.receive("prediction")
        line = json.dumps(prediction, ensure_ascii=False)
        if self.output_handle is not None:
            self.output_handle.write(line + "\n")
            self.output_handle.flush()
        print(line, flush=True)
        if not self.headless:
            cv2.imshow("Holoscan Cyclist Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def stop(self):
        if self.output_handle is not None:
            self.output_handle.close()
        if not self.headless:
            cv2.destroyAllWindows()


class CyclistPredictionApplication(Application):
    def __init__(self, runtime_args):
        super().__init__()
        self.runtime_args = runtime_args

    def compose(self):
        if self.runtime_args.camera == "synthetic":
            source = SyntheticCameraOp(
                self,
                CountCondition(
                    self,
                    max(
                        1,
                        int(round(self.runtime_args.duration * self.runtime_args.fps)),
                    ),
                ),
                name="synthetic_source",
                width=self.runtime_args.width,
                height=self.runtime_args.height,
                fps=self.runtime_args.fps,
                duration=self.runtime_args.duration,
            )
        else:
            source = CameraOp(
                self,
                name="camera",
                index=self.runtime_args.webcam_index,
                width=self.runtime_args.width,
                height=self.runtime_args.height,
                fps=self.runtime_args.fps,
                duration=self.runtime_args.duration,
            )
        inference = CyclistPredictionOp(self, name="cyclist_prediction", args=self.runtime_args)
        display = OutputOp(
            self,
            name="output",
            headless=self.runtime_args.headless,
            output_jsonl=self.runtime_args.output_jsonl,
        )
        self.add_flow(source, inference, {("frame", "frame")})
        self.add_flow(inference, display,
                      {("annotated", "frame"), ("prediction", "prediction")})


def runtime_args(cli):
    return SimpleNamespace(
        model=cli.model, pose=cli.pose, camera=cli.camera, webcam_index=cli.webcam_index,
        width=cli.width, height=cli.height, fps=cli.fps, device=cli.device,
        duration=cli.duration, headless=cli.headless,
        output_jsonl=cli.output_jsonl,
        torch_threads=cli.torch_threads,
        bus_swap_upper_labels=False, intersection_fold=5,
        bus_config=None, bus_head_model=None, bus_upper_model=None,
        bus_leg_model=None, bus_stage2_model=None,
        intersection_explicit_model=None, intersection_implicit_model=None,
        intersection_maneuver_model=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("bus", "intersection", "synthetic"),
        default="bus",
    )
    parser.add_argument("--camera", choices=("webcam", "synthetic"), default="webcam")
    parser.add_argument("--pose", choices=("mediapipe", "trt", "synthetic"))
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    cli = parser.parse_args()
    if cli.torch_threads < 1:
        parser.error("--torch-threads must be at least 1")
    if cli.pose is None:
        cli.pose = "synthetic" if cli.camera == "synthetic" else "mediapipe"
    if cli.camera == "synthetic" and cli.duration <= 0:
        cli.duration = 10.0
    if cli.headless and cli.output_jsonl is None:
        cli.output_jsonl = Path("holoscan_predictions.jsonl")
    try:
        CyclistPredictionApplication(runtime_args(cli)).run()
    except KeyboardInterrupt:
        print("Holoscan graph stopped.", flush=True)


if __name__ == "__main__":
    main()
