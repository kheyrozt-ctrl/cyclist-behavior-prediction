"""Camera sources shared by all real-time prediction pipelines."""

import cv2
import numpy as np


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


def create_camera(source, width=640, height=480, fps=30, webcam_index=0):
    if source == "realsense":
        return RealSenseSource(width=width, height=height, fps=fps)
    if source == "webcam":
        return WebcamSource(index=webcam_index, width=width, height=height, fps=fps)
    raise ValueError(f"Unknown camera source: {source}")

