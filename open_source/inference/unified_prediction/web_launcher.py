#!/usr/bin/env python3
"""Dependency-free web launcher and MJPEG preview for unified prediction."""

import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")


class PipelineController:
    def __init__(self):
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.stop_event = None
        self.thread = None
        self.frame = None
        self.frame_id = 0
        self.state = "idle"
        self.error = None
        self.metrics = {
            "fps": 0, "pose_ok": False, "frame_size": None,
            "predictions": [], "progress": None,
        }
        self.started_at = None
        self.config = None

    def snapshot(self):
        with self.lock:
            return {
                "state": self.state,
                "error": self.error,
                "metrics": self.metrics,
                "started_at": self.started_at,
                "config": self.config,
            }

    def start(self, config):
        with self.lock:
            if self.state in ("starting", "running", "stopping"):
                raise RuntimeError("Pipeline is already active")
            args, safe_config = build_args(config)
            self.state = "starting"
            self.error = None
            self.frame = None
            self.metrics = {
                "fps": 0, "pose_ok": False, "frame_size": None,
                "predictions": [], "progress": None,
            }
            self.config = safe_config
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._run, args=(args,), daemon=True)
            self.thread.start()

    def _run(self, args):
        try:
            # Heavy ML/video dependencies are intentionally loaded only when a
            # pipeline starts, so the console can open and report setup errors.
            from .predictors import create_predictor
            from .runtime import run_realtime

            predictor = create_predictor(args)
            with self.lock:
                self.state = "running"
                self.started_at = time.time()
            run_realtime(
                args,
                predictor,
                stop_event=self.stop_event,
                frame_callback=self._on_frame,
                status_callback=self._on_status,
            )
        except ModuleNotFoundError as exc:
            with self.lock:
                package = exc.name or "unknown"
                self.error = (
                    f"Missing Python package '{package}'. Close this server and "
                    "start it with start_unified_web.bat or start_unified_web.sh "
                    "to install the required environment."
                )
                self.state = "error"
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.state = "error"
        finally:
            with self.lock:
                if self.state != "error":
                    self.state = "idle"
                self.started_at = None
                self.frame_ready.notify_all()

    def _on_frame(self, frame):
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            return
        with self.frame_ready:
            self.frame = encoded.tobytes()
            self.frame_id += 1
            self.frame_ready.notify_all()

    def _on_status(self, metrics):
        with self.lock:
            self.metrics = metrics

    def stop(self):
        with self.lock:
            if self.state not in ("starting", "running"):
                return
            self.state = "stopping"
            self.stop_event.set()

    def wait_for_frame(self, last_id):
        with self.frame_ready:
            self.frame_ready.wait_for(
                lambda: self.frame_id != last_id,
                timeout=2.0,
            )
            return self.frame_id, self.frame

    def latest_frame(self):
        with self.lock:
            return self.frame_id, self.frame


def integer(value, name, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def build_args(raw):
    model = raw.get("model", "bus")
    pose = raw.get("pose", "mediapipe")
    camera = raw.get("camera", "realsense")
    device = str(raw.get("device", "")).strip() or None
    if model not in ("bus", "intersection"):
        raise ValueError("Unknown model")
    if pose not in ("mediapipe", "trt"):
        raise ValueError("Unknown pose backend")
    if camera not in ("realsense", "webcam"):
        raise ValueError("Unknown camera source")

    safe = {
        "model": model,
        "pose": pose,
        "camera": camera,
        "webcam_index": integer(raw.get("webcam_index", 0), "Webcam index", 0, 32),
        "width": integer(raw.get("width", 640), "Width", 320, 3840),
        "height": integer(raw.get("height", 480), "Height", 240, 2160),
        "fps": integer(raw.get("fps", 30), "FPS", 1, 120),
        "intersection_fold": integer(raw.get("intersection_fold", 5), "Fold", 1, 5),
        "device": device,
        "bus_swap_upper_labels": bool(raw.get("bus_swap_upper_labels", False)),
    }
    return argparse.Namespace(
        **safe,
        headless=False,
        output=None,
        duration=None,
        web_stream=True,
        bus_config=None,
        bus_head_model=None,
        bus_upper_model=None,
        bus_leg_model=None,
        bus_stage2_model=None,
        intersection_explicit_model=None,
        intersection_implicit_model=None,
        intersection_maneuver_model=None,
    ), safe


class RequestHandler(SimpleHTTPRequestHandler):
    controller = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        # The launcher is a local development-style console; stale HTML/CSS can
        # otherwise preserve old icons after an update.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/api/status", "/api/frame")):
            super().log_message(fmt, *args)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return self.send_json(200, self.controller.snapshot())
        if path == "/api/frame":
            return self.send_latest_frame()
        if path == "/api/stream":
            return self.stream_frames()
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def send_latest_frame(self):
        frame_id, frame = self.controller.latest_frame()
        if frame is None:
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Frame-Id", str(frame_id))
        self.end_headers()
        self.wfile.write(frame)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/start", "/api/stop"):
            return self.send_json(404, {"error": "Not found"})
        try:
            if path == "/api/start":
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024:
                    raise ValueError("Request is too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                self.controller.start(payload)
            else:
                self.controller.stop()
            return self.send_json(202, self.controller.snapshot())
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": str(exc)})

    def stream_frames(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_id = -1
        try:
            while True:
                frame_id, frame = self.controller.wait_for_frame(last_id)
                if frame is None or frame_id == last_id:
                    continue
                last_id = frame_id
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description="Unified prediction web launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    controller = PipelineController()
    RequestHandler.controller = controller
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"Unified web launcher: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web launcher...")
    finally:
        controller.stop()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
