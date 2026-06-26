"""Shared real-time loop for cyclist prediction models."""

import os
import sys
import time

import cv2

from .camera import create_camera


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CYCLIST_INFERENCE_DIR = os.path.join(ROOT_DIR, "cyclist_inference")
if CYCLIST_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, CYCLIST_INFERENCE_DIR)

from pose_adapter import create_pose_adapter  # noqa: E402


def draw_text(frame, text, pos, color, scale=0.65):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def draw_overlay(frame, title, fps, pose_ok, lines, progress=None):
    status = "pose ok" if pose_ok else "no pose"
    all_lines = [
        (title, (255, 255, 255)),
        (f"FPS: {fps:.1f}  {status}", (0, 255, 0) if pose_ok else (0, 255, 255)),
    ]
    all_lines.extend(lines)

    for i, (text, color) in enumerate(all_lines):
        draw_text(frame, text, (10, 30 + i * 28), color)

    if progress is None:
        return frame

    current, total = progress
    ratio = min(1.0, current / total) if total else 0.0
    bar_y = 30 + len(all_lines) * 28 + 8
    bar_width = 320
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 14), (45, 45, 45), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(bar_width * ratio), bar_y + 14),
                  (0, 210, 255), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + 14), (255, 255, 255), 1)
    return frame


def run_realtime(args, predictor, stop_event=None, frame_callback=None, status_callback=None):
    """Run the capture loop.

    Optional callbacks are used by the web launcher. Existing CLI callers keep
    the same behavior when they are omitted.
    """
    camera = create_camera(
        args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        webcam_index=args.webcam_index,
    )
    pose = create_pose_adapter(args.pose)

    writer = None
    if args.headless:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, float(args.fps),
                                 (args.width, args.height))
        print(f"Headless mode: recording to {args.output}")
    elif not getattr(args, "web_stream", False):
        print("Starting unified prediction. Press 'q' to quit.")

    frame_count = 0
    fps = 0.0
    fps_start = time.time()
    run_start = time.time()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
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
                predictor.update(keypoints, now)

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            pose.draw_skeleton(frame)
            draw_overlay(
                frame,
                predictor.title,
                fps,
                pose_ok,
                predictor.overlay_lines(),
                predictor.progress(),
            )

            if frame_callback is not None:
                frame_callback(frame)
            if status_callback is not None:
                status_callback({
                    "fps": round(fps, 1),
                    "pose_ok": pose_ok,
                    "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                    "predictions": [text for text, _color in predictor.overlay_lines()],
                    "progress": predictor.progress(),
                })

            if writer:
                writer.write(frame)
            elif not getattr(args, "web_stream", False):
                cv2.imshow("Unified Cyclist Prediction", frame)
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
        if not args.headless and not getattr(args, "web_stream", False):
            cv2.destroyAllWindows()
        print("Pipeline stopped.")
