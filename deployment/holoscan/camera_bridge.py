#!/usr/bin/env python3
"""Expose a host webcam as a local MJPEG stream for Docker Desktop."""

import argparse
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class CameraStream:
    def __init__(self, index, width, height, fps, jpeg_quality):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.capture = None
        self.lock = threading.Lock()

    def open(self):
        capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open host webcam index {self.index}")
        self.capture = capture

    def jpeg(self):
        with self.lock:
            ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Host webcam frame acquisition failed")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class CameraBridgeHandler(BaseHTTPRequestHandler):
    stream = None

    def log_message(self, fmt, *args):
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                frame = self.stream.jpeg()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    stream = CameraStream(
        args.webcam_index,
        args.width,
        args.height,
        args.fps,
        args.jpeg_quality,
    )
    stream.open()
    CameraBridgeHandler.stream = stream
    server = ThreadingHTTPServer((args.host, args.port), CameraBridgeHandler)
    signal.signal(
        signal.SIGTERM,
        lambda *_: threading.Thread(target=server.shutdown, daemon=True).start(),
    )
    print(
        f"Camera bridge ready: http://{args.host}:{args.port}/stream.mjpg",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stream.close()


if __name__ == "__main__":
    main()
