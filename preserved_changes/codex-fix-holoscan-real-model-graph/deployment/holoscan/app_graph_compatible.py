#!/usr/bin/env python3
"""
Holoscan-style cyclist prediction graph for JetPack 5.x.

This version does not require the NVIDIA Holoscan SDK.
It keeps the graph structure:

CameraSourceOp
    -> PoseEstimationOp
    -> CyclistPredictorOp
    -> OverlayOp
    -> OutputOp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


class CameraSourceOp:
    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps: int,
        webcam_index: int = 0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.webcam_index = webcam_index
        self.camera = None
        self.frame_index = 0

    def start(self):
        if self.source == "synthetic":
            self.camera = None
            print("[CameraSourceOp] Using synthetic camera")
            return

        from unified_prediction.camera import create_camera

        print(f"[CameraSourceOp] Opening camera source: {self.source}")
        self.camera = create_camera(
            self.source,
            width=self.width,
            height=self.height,
            fps=self.fps,
            webcam_index=self.webcam_index,
        )

    def compute(self) -> Optional[Dict[str, Any]]:
        if self.source == "synthetic":
            frame = self._make_synthetic_frame()
        else:
            frame = self.camera.read()
            if frame is None:
                print("[CameraSourceOp] Warning: empty frame")
                return None

        packet = {
            "frame": frame,
            "timestamp": time.time(),
            "frame_index": self.frame_index,
            "source": self.source,
        }

        self.frame_index += 1
        return packet

    def _make_synthetic_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        x = int((self.frame_index * 8) % max(1, self.width))
        y = self.height // 2

        cv2.circle(frame, (x, y), 28, (0, 255, 0), -1)
        cv2.putText(
            frame,
            "Synthetic cyclist frame",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return frame

    def stop(self):
        if self.camera is not None:
            print("[CameraSourceOp] Closing camera")
            self.camera.close()


class PoseEstimationOp:
    def __init__(self, pose_backend: str):
        self.pose_backend = pose_backend
        self.pose = None

    def start(self):
        if self.pose_backend == "synthetic":
            self.pose = None
            print("[PoseEstimationOp] Using synthetic pose")
            return

        from cyclist_inference.pose_adapter import create_pose_adapter

        print(f"[PoseEstimationOp] Loading pose backend: {self.pose_backend}")
        self.pose = create_pose_adapter(self.pose_backend)

    def compute(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        frame = packet["frame"]

        if self.pose_backend == "synthetic":
            keypoints = self._synthetic_keypoints()
            pose_ok = True
        else:
            self.pose.process(frame)
            keypoints = self.pose.extract_keypoints()
            pose_ok = keypoints is not None

            if pose_ok:
                self.pose.draw_skeleton(frame)

        return {
            **packet,
            "keypoints": keypoints,
            "pose_ok": pose_ok,
        }

    def _synthetic_keypoints(self) -> np.ndarray:
        points = []

        for i in range(33):
            x = 0.5 + 0.2 * np.sin(i * 0.4)
            y = 0.5 + 0.3 * np.cos(i * 0.3)
            z = 0.0
            visibility = 1.0
            points.append([x, y, z, visibility])

        return np.asarray(points, dtype=np.float32)

    def stop(self):
        if self.pose is not None:
            print("[PoseEstimationOp] Closing pose backend")
            self.pose.close()


class CyclistPredictorOp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.predictor = None

    def start(self):
        if self.args.model == "synthetic":
            print("[CyclistPredictorOp] Using synthetic predictor")
            self.predictor = SyntheticPredictor()
            return

        import torch

        torch.set_num_threads(self.args.torch_threads)

        from unified_prediction.predictors import create_predictor

        print(f"[CyclistPredictorOp] Loading model: {self.args.model}")
        self.predictor = create_predictor(self.args)

    def compute(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        keypoints = packet["keypoints"]
        timestamp = packet["timestamp"]

        if packet["pose_ok"] and keypoints is not None:
            self.predictor.update(keypoints, timestamp)

        lines = self.predictor.overlay_lines()
        progress = self.predictor.progress()

        return {
            **packet,
            "title": getattr(self.predictor, "title", "Cyclist Prediction"),
            "lines": lines,
            "progress": progress,
        }


class SyntheticPredictor:
    title = "Synthetic Cyclist Prediction"

    def __init__(self):
        self.counter = 0
        self.labels = ["straight", "left", "right"]

    def update(self, keypoints, timestamp):
        self.counter += 1

    def overlay_lines(self):
        label = self.labels[(self.counter // 30) % len(self.labels)]

        return [
            (f"Prediction: {label}", (0, 255, 0)),
            (f"Frames: {self.counter}", (255, 255, 255)),
        ]

    def progress(self):
        return self.counter, 30


class OverlayOp:
    def compute(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        from unified_prediction.runtime import draw_overlay

        frame = packet["frame"]
        lines = packet["lines"]
        progress = packet["progress"]

        draw_overlay(
            frame,
            packet["title"],
            0.0,
            packet["pose_ok"],
            lines,
            progress,
        )

        prediction = {
            "frame_index": packet["frame_index"],
            "timestamp": packet["timestamp"],
            "source": packet["source"],
            "pose_ok": packet["pose_ok"],
            "progress": progress,
            "predictions": [text for text, _color in lines],
        }

        return {
            "frame": frame,
            "prediction": prediction,
        }


class OutputOp:
    def __init__(
        self,
        headless: bool,
        output_jsonl: Optional[str],
        output_video: Optional[str],
        fps: int,
        width: int,
        height: int,
    ):
        self.headless = headless
        self.output_jsonl = output_jsonl
        self.output_video = output_video
        self.fps = fps
        self.width = width
        self.height = height
        self.jsonl_file = None
        self.video_writer = None

    def start(self):
        if self.output_jsonl:
            print(f"[OutputOp] Writing JSONL to {self.output_jsonl}")
            self.jsonl_file = open(self.output_jsonl, "w", encoding="utf-8")

        if self.output_video:
            print(f"[OutputOp] Writing video to {self.output_video}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.output_video,
                fourcc,
                float(self.fps),
                (self.width, self.height),
            )

    def compute(self, packet: Dict[str, Any]):
        frame = packet["frame"]
        prediction = packet["prediction"]

        if self.jsonl_file:
            self.jsonl_file.write(json.dumps(prediction) + "\n")
            self.jsonl_file.flush()

        if self.video_writer:
            self.video_writer.write(frame)

        if not self.headless:
            cv2.imshow("Cyclist Holoscan-style Graph", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                raise KeyboardInterrupt

    def stop(self):
        if self.jsonl_file:
            self.jsonl_file.close()

        if self.video_writer:
            self.video_writer.release()

        if not self.headless:
            cv2.destroyAllWindows()


class CyclistPredictionGraph:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.camera = CameraSourceOp(
            source=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            webcam_index=args.webcam_index,
        )

        self.pose = PoseEstimationOp(
            pose_backend=args.pose,
        )

        self.predictor = CyclistPredictorOp(args)

        self.overlay = OverlayOp()

        self.output = OutputOp(
            headless=args.headless,
            output_jsonl=args.output_jsonl,
            output_video=args.output_video,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )

    def start(self):
        print("=" * 70)
        print("Starting Holoscan-style cyclist prediction graph")
        print("CameraSourceOp -> PoseEstimationOp -> CyclistPredictorOp -> OverlayOp -> OutputOp")
        print("=" * 70)

        self.camera.start()
        self.pose.start()
        self.predictor.start()
        self.output.start()

    def run(self):
        self.start()
        start_time = time.monotonic()

        try:
            while True:
                if self.args.duration > 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= self.args.duration:
                        break

                frame_packet = self.camera.compute()
                if frame_packet is None:
                    continue

                pose_packet = self.pose.compute(frame_packet)
                prediction_packet = self.predictor.compute(pose_packet)
                output_packet = self.overlay.compute(prediction_packet)
                self.output.compute(output_packet)

        except KeyboardInterrupt:
            print("Stopped by user.")

        finally:
            self.stop()

    def stop(self):
        print("[Graph] Stopping operators")
        self.output.stop()
        self.pose.stop()
        self.camera.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Holoscan-style cyclist behavior prediction graph"
    )

    parser.add_argument(
        "--camera",
        choices=["realsense", "webcam", "synthetic"],
        default="realsense",
    )
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--pose",
        choices=["trt", "mediapipe", "synthetic"],
        default="mediapipe",
    )
    parser.add_argument(
        "--model",
        choices=["intersection", "bus", "synthetic"],
        default="intersection",
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=1)

    parser.add_argument("--intersection-fold", type=int, default=0)
    parser.add_argument("--intersection-explicit-model", default=None)
    parser.add_argument("--intersection-implicit-model", default=None)
    parser.add_argument("--intersection-maneuver-model", default=None)

    parser.add_argument("--bus-model-path", default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-video", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    graph = CyclistPredictionGraph(args)
    graph.run()


if __name__ == "__main__":
    main()