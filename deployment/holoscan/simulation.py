"""Deterministic synthetic inputs for headless Holoscan graph validation."""

from __future__ import annotations

import math

import cv2
import numpy as np


_POSE_TEMPLATE = (
    (0.50, 0.14), (0.48, 0.13), (0.47, 0.13), (0.45, 0.14),
    (0.52, 0.13), (0.53, 0.13), (0.55, 0.14), (0.43, 0.16),
    (0.57, 0.16), (0.47, 0.19), (0.53, 0.19), (0.42, 0.30),
    (0.58, 0.30), (0.37, 0.43), (0.63, 0.43), (0.34, 0.57),
    (0.66, 0.57), (0.33, 0.59), (0.67, 0.59), (0.33, 0.58),
    (0.67, 0.58), (0.34, 0.56), (0.66, 0.56), (0.45, 0.56),
    (0.55, 0.56), (0.42, 0.72), (0.58, 0.72), (0.39, 0.87),
    (0.61, 0.87), (0.38, 0.89), (0.62, 0.89), (0.41, 0.91),
    (0.59, 0.91),
)


def synthetic_keypoints(frame_index: int, fps: float) -> list[float]:
    """Return 33 normalized 2D landmarks with a deterministic cycling motion."""
    phase = 2.0 * math.pi * frame_index / max(float(fps), 1.0)
    output: list[float] = []
    for index, (base_x, base_y) in enumerate(_POSE_TEMPLATE):
        upper_motion = 0.012 * math.sin(phase + index * 0.17) if index < 23 else 0.0
        pedal_motion = 0.018 * math.sin(phase * 1.8 + index) if index >= 25 else 0.0
        output.extend(
            [
                float(np.clip(base_x + upper_motion, 0.0, 1.0)),
                float(np.clip(base_y + pedal_motion, 0.0, 1.0)),
            ]
        )
    return output


def synthetic_frame(
    frame_index: int,
    width: int,
    height: int,
    fps: float,
) -> np.ndarray:
    """Create a deterministic road-style frame for graph transport testing."""
    frame = np.full((height, width, 3), (24, 31, 38), dtype=np.uint8)
    horizon = int(height * 0.42)
    cv2.rectangle(frame, (0, horizon), (width, height), (44, 50, 57), -1)
    cv2.line(
        frame,
        (width // 2, horizon),
        (width // 2, height),
        (62, 183, 219),
        max(2, width // 240),
    )
    cv2.putText(
        frame,
        f"SYNTHETIC FRAME {frame_index:06d}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 240, 245),
        2,
        cv2.LINE_AA,
    )
    return frame
