"""Pose estimation adapter — unified interface for MediaPipe and trt_pose backends.

Both adapters produce 66 features (33 keypoints x 2 coords) compatible with
the Bi-LSTM gesture classifiers trained on MediaPipe data.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional


class PoseAdapter(ABC):
    """Abstract base class for pose estimation backends."""

    @abstractmethod
    def process(self, bgr_frame) -> None:
        """Run pose estimation on a BGR frame."""

    @abstractmethod
    def extract_keypoints(self) -> Optional[List[float]]:
        """Return 66 floats (33 keypoints x 2 coords) or None if no pose detected."""

    @abstractmethod
    def draw_skeleton(self, frame) -> None:
        """Draw pose skeleton overlay on the frame (in-place)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class MediaPipeAdapter(PoseAdapter):
    """Wraps MediaPipe BlazePose (33 keypoints)."""

    def __init__(
        self,
        model_complexity=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
        input_width=None,
    ):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe build does not provide the Solutions API required "
                "by BlazePose. Install the compatible version with: "
                "python -m pip install mediapipe==0.10.21"
            )
        self._mp = mp
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._input_width = input_width
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._results = None

    def process(self, bgr_frame) -> None:
        import cv2
        inference_frame = bgr_frame
        if self._input_width and bgr_frame.shape[1] > self._input_width:
            height = int(round(
                bgr_frame.shape[0] * self._input_width / bgr_frame.shape[1]
            ))
            inference_frame = cv2.resize(
                bgr_frame,
                (self._input_width, height),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        self._results = self._pose.process(rgb)

    def extract_keypoints(self) -> Optional[List[float]]:
        if self._results is None or not self._results.pose_landmarks:
            return None
        keypoints = []
        for lm in self._results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
        return keypoints

    def draw_skeleton(self, frame) -> None:
        if self._results is None or not self._results.pose_landmarks:
            return
        self._mp_drawing.draw_landmarks(
            frame,
            self._results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            self._mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self._mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
        )

    def close(self) -> None:
        self._pose.close()


class TrtPoseAdapter(PoseAdapter):
    """Wraps trt_pose (18 COCO+neck keypoints -> mapped to 33 MediaPipe keypoints).

    Imports from pose_detection/pose_detect.py for model loading, preprocessing,
    and skeleton drawing.
    """

    # Minimum number of detected trt_pose keypoints to produce a valid output.
    MIN_KEYPOINTS = 5

    def __init__(self):
        # Add pose_detection to path so we can import pose_detect
        pose_det_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "pose_detection"
        )
        if pose_det_dir not in sys.path:
            sys.path.insert(0, pose_det_dir)

        from pose_detect import load_trt_pose_model, preprocess_frame, draw_skeleton
        self._preprocess_frame = preprocess_frame
        self._draw_skeleton_fn = draw_skeleton

        import torch
        self._torch = torch

        print("Loading trt_pose model...")
        self._model, self._parse_objects, self._topology, self._human_pose = (
            load_trt_pose_model()
        )

        # Inference results (set by process())
        self._counts = None
        self._objects = None
        self._peaks = None

    def process(self, bgr_frame) -> None:
        tensor = self._preprocess_frame(bgr_frame)
        with self._torch.no_grad():
            cmap, paf = self._model(tensor)
        cmap = cmap.detach().cpu()
        paf = paf.detach().cpu()
        self._counts, self._objects, self._peaks = self._parse_objects(cmap, paf)

    def extract_keypoints(self) -> Optional[List[float]]:
        if self._counts is None or int(self._counts[0]) == 0:
            return None

        # Extract raw trt_pose keypoints for the first detected person
        obj = self._objects[0][0]
        num_parts = len(self._human_pose["keypoints"])  # 18
        raw = {}  # trt_index -> (x, y) normalized
        for j in range(num_parts):
            k = int(obj[j])
            if k >= 0:
                peak = self._peaks[0][j][k]
                # peaks are (row, col) = (y, x)
                raw[j] = (float(peak[1]), float(peak[0]))

        if len(raw) < self.MIN_KEYPOINTS:
            return None

        return self._trt_to_mediapipe_keypoints(raw)

    def draw_skeleton(self, frame) -> None:
        if self._counts is None:
            return
        self._draw_skeleton_fn(
            frame, self._counts, self._objects, self._peaks,
            self._topology, self._human_pose,
        )

    def close(self) -> None:
        pass  # TensorRT model doesn't need explicit cleanup

    @staticmethod
    def _trt_to_mediapipe_keypoints(raw: dict) -> List[float]:
        """Map 18 trt_pose keypoints to 33 MediaPipe keypoints (66 floats).

        Mapping strategies:
          - Direct: trt_pose keypoint maps 1:1 to MediaPipe keypoint.
          - Duplicate: MediaPipe keypoint copies a nearby trt_pose keypoint
            (e.g. left_eye -> left_eye_inner/outer).
          - Interpolate: Synthetic point computed from weighted combination
            (mouth points from nose + neck + lateral eye shift).

        Missing keypoints default to (0.0, 0.0).
        """
        def get(idx):
            return raw.get(idx, (0.0, 0.0))

        mp = [None] * 33

        # 0: nose <- trt 0
        mp[0] = get(0)
        # 1-3: left eye variants <- trt 1 (left_eye)
        mp[1] = get(1)   # left_eye_inner
        mp[2] = get(1)   # left_eye
        mp[3] = get(1)   # left_eye_outer
        # 4-6: right eye variants <- trt 2 (right_eye)
        mp[4] = get(2)   # right_eye_inner
        mp[5] = get(2)   # right_eye
        mp[6] = get(2)   # right_eye_outer
        # 7-8: ears
        mp[7] = get(3)   # left_ear
        mp[8] = get(4)   # right_ear

        # 9: mouth_left — interpolate(70% nose + 30% neck), shift 15% toward left_eye
        nose = get(0)
        neck = get(17)
        left_eye = get(1)
        right_eye = get(2)
        mouth_base_x = 0.7 * nose[0] + 0.3 * neck[0]
        mouth_base_y = 0.7 * nose[1] + 0.3 * neck[1]
        mp[9] = (
            mouth_base_x + 0.15 * (left_eye[0] - mouth_base_x),
            mouth_base_y + 0.15 * (left_eye[1] - mouth_base_y),
        )
        # 10: mouth_right — same base, shift 15% toward right_eye
        mp[10] = (
            mouth_base_x + 0.15 * (right_eye[0] - mouth_base_x),
            mouth_base_y + 0.15 * (right_eye[1] - mouth_base_y),
        )

        # 11-16: upper body direct mappings
        mp[11] = get(5)   # left_shoulder
        mp[12] = get(6)   # right_shoulder
        mp[13] = get(7)   # left_elbow
        mp[14] = get(8)   # right_elbow
        mp[15] = get(9)   # left_wrist
        mp[16] = get(10)  # right_wrist

        # 17-22: hand keypoints — duplicate from wrists
        mp[17] = get(9)   # left_pinky <- left_wrist
        mp[18] = get(10)  # right_pinky <- right_wrist
        mp[19] = get(9)   # left_index <- left_wrist
        mp[20] = get(10)  # right_index <- right_wrist
        mp[21] = get(9)   # left_thumb <- left_wrist
        mp[22] = get(10)  # right_thumb <- right_wrist

        # 23-28: lower body direct mappings
        mp[23] = get(11)  # left_hip
        mp[24] = get(12)  # right_hip
        mp[25] = get(13)  # left_knee
        mp[26] = get(14)  # right_knee
        mp[27] = get(15)  # left_ankle
        mp[28] = get(16)  # right_ankle

        # 29-32: foot keypoints — duplicate from ankles
        mp[29] = get(15)  # left_heel <- left_ankle
        mp[30] = get(16)  # right_heel <- right_ankle
        mp[31] = get(15)  # left_foot_index <- left_ankle
        mp[32] = get(16)  # right_foot_index <- right_ankle

        # Flatten to 66 floats: [x0, y0, x1, y1, ...]
        keypoints = []
        for x, y in mp:
            keypoints.extend([x, y])
        return keypoints


def create_pose_adapter(backend: str, **kwargs) -> PoseAdapter:
    """Factory function to create a pose estimation adapter.

    Args:
        backend: "mediapipe" or "trt"

    Returns:
        PoseAdapter instance
    """
    if backend == "mediapipe":
        return MediaPipeAdapter(**kwargs)
    elif backend == "trt":
        return TrtPoseAdapter()
    else:
        raise ValueError(
            f"Unknown pose backend: {backend!r}. Choose 'mediapipe' or 'trt'."
        )
