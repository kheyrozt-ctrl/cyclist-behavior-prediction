#!/usr/bin/env python3
"""Holoscan graph for camera -> cyclist prediction -> display/publication."""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if sys.platform == "win32":
    raise SystemExit(
        "NVIDIA Holoscan SDK is not available as a native Windows Python "
        "runtime. Run this graph on a supported Linux x86_64/CUDA or Jetson "
        "environment. On Windows, use: "
        r".venv\Scripts\python.exe unified_prediction\run_web.py"
    )

try:
    from holoscan.core import Application, Operator, OperatorSpec
except ModuleNotFoundError as exc:
    if exc.name != "holoscan":
        raise
    raise SystemExit(
        "Holoscan SDK is not installed in this interpreter. On supported "
        "Linux/Jetson systems run: "
        "python3 -m pip install -r deployment/holoscan/requirements.txt"
    ) from exc


class CameraOp(Operator):
    def __init__(self, fragment, *, index=0, width=640, height=480, fps=30, **kwargs):
        super().__init__(fragment, **kwargs)
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.capture = None

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.capture = cv2.VideoCapture(self.index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {self.index}")

    def compute(self, op_input, op_output, context):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera frame acquisition failed")
        op_output.emit(frame, "frame")

    def stop(self):
        if self.capture is not None:
            self.capture.release()


class CyclistPredictionOp(Operator):
    def __init__(self, fragment, *, args, **kwargs):
        super().__init__(fragment, **kwargs)
        self.args = args
        self.pose = self.predictor = None

    def setup(self, spec: OperatorSpec):
        spec.input("frame")
        spec.output("annotated")
        spec.output("prediction")

    def start(self):
        from cyclist_inference.pose_adapter import create_pose_adapter
        from unified_prediction.predictors import create_predictor
        self.pose = create_pose_adapter(self.args.pose)
        self.predictor = create_predictor(self.args)

    def compute(self, op_input, op_output, context):
        from unified_prediction.runtime import draw_overlay
        frame = op_input.receive("frame")
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
        op_output.emit({"timestamp": now, "pose_ok": keypoints is not None,
                        "lines": [text for text, _ in lines]}, "prediction")

    def stop(self):
        if self.pose is not None:
            self.pose.close()


class DisplayOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("frame")
        spec.input("prediction")

    def compute(self, op_input, op_output, context):
        frame = op_input.receive("frame")
        prediction = op_input.receive("prediction")
        cv2.imshow("Holoscan Cyclist Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
        print(prediction, flush=True)

    def stop(self):
        cv2.destroyAllWindows()


class CyclistPredictionApplication(Application):
    def __init__(self, runtime_args):
        super().__init__()
        self.runtime_args = runtime_args

    def compose(self):
        source = CameraOp(self, name="camera", index=self.runtime_args.webcam_index,
                          width=self.runtime_args.width, height=self.runtime_args.height,
                          fps=self.runtime_args.fps)
        inference = CyclistPredictionOp(self, name="cyclist_inference", args=self.runtime_args)
        display = DisplayOp(self, name="display")
        self.add_flow(source, inference, {("frame", "frame")})
        self.add_flow(inference, display,
                      {("annotated", "frame"), ("prediction", "prediction")})


def runtime_args(cli):
    return SimpleNamespace(
        model=cli.model, pose=cli.pose, camera="webcam", webcam_index=cli.webcam_index,
        width=cli.width, height=cli.height, fps=cli.fps, device=cli.device,
        bus_swap_upper_labels=False, intersection_fold=5,
        bus_config=None, bus_head_model=None, bus_upper_model=None,
        bus_leg_model=None, bus_stage2_model=None,
        intersection_explicit_model=None, intersection_implicit_model=None,
        intersection_maneuver_model=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bus", "intersection"), default="bus")
    parser.add_argument("--pose", choices=("mediapipe", "trt"), default="mediapipe")
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device")
    cli = parser.parse_args()
    CyclistPredictionApplication(runtime_args(cli)).run()


if __name__ == "__main__":
    main()
