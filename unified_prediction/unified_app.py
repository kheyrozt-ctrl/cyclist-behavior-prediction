#!/usr/bin/env python3
"""Single unified app for bus-stop and intersection cyclist prediction."""

import argparse
import copy

from .predictors import create_predictor
from .runtime import run_realtime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified cyclist behavior prediction app"
    )
    parser.add_argument(
        "--model",
        choices=["bus", "intersection"],
        help="Model pipeline to run. If omitted, a GUI selector is shown.",
    )
    parser.add_argument(
        "--pose",
        choices=["mediapipe", "trt"],
        default="mediapipe",
        help="Pose backend shared by both model pipelines.",
    )
    parser.add_argument(
        "--camera",
        choices=["realsense", "webcam"],
        default="realsense",
        help="Camera source shared by both model pipelines.",
    )
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--device")

    # Bus model options
    parser.add_argument("--bus-config")
    parser.add_argument("--bus-head-model")
    parser.add_argument("--bus-upper-model")
    parser.add_argument("--bus-leg-model")
    parser.add_argument("--bus-stage2-model")
    parser.add_argument("--bus-swap-upper-labels", action="store_true")

    # Intersection model options
    parser.add_argument("--intersection-fold", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--intersection-explicit-model")
    parser.add_argument("--intersection-implicit-model")
    parser.add_argument("--intersection-maneuver-model")
    return parser.parse_args()


def choose_model_cli():
    print("")
    print("Select cyclist prediction model:")
    print("  1) Bus-stop behavior: straight / yield / overtake")
    print("  2) Intersection intention: Crossing / LeftTurn / RightTurn")
    print("")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "bus"
        if choice == "2":
            return "intersection"
        print("Invalid choice. Please enter 1 or 2.")


def choose_options_gui(args):
    import tkinter as tk
    from tkinter import ttk

    selected = {"args": None}

    root = tk.Tk()
    root.title("Unified Cyclist Prediction")
    root.resizable(False, False)
    root.geometry("560x500")

    model_var = tk.StringVar(value="bus")
    pose_var = tk.StringVar(value=args.pose)
    camera_var = tk.StringVar(value=args.camera)
    headless_var = tk.BooleanVar(value=args.headless)
    swap_upper_var = tk.BooleanVar(value=args.bus_swap_upper_labels)
    duration_var = tk.StringVar(value="" if args.duration is None else str(args.duration))
    output_var = tk.StringVar(value=args.output or "")
    fold_var = tk.IntVar(value=args.intersection_fold)
    device_var = tk.StringVar(value=args.device or "")

    main = ttk.Frame(root, padding=16)
    main.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(0, weight=1)

    ttk.Label(
        main,
        text="Unified Cyclist Prediction",
        font=("TkDefaultFont", 14, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 12))

    model_frame = ttk.LabelFrame(main, text="Model")
    model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    model_frame.columnconfigure(0, weight=1)
    ttk.Radiobutton(
        model_frame,
        text="Bus-stop behavior: straight / yield / overtake",
        variable=model_var,
        value="bus",
    ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))
    ttk.Radiobutton(
        model_frame,
        text="Intersection intention: Crossing / LeftTurn / RightTurn",
        variable=model_var,
        value="intersection",
    ).grid(row=1, column=0, sticky="w", padx=10, pady=(4, 8))

    runtime_frame = ttk.LabelFrame(main, text="Shared Runtime")
    runtime_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    runtime_frame.columnconfigure(1, weight=1)
    ttk.Label(runtime_frame, text="Pose backend").grid(row=0, column=0, sticky="w", padx=10, pady=6)
    ttk.Combobox(
        runtime_frame,
        textvariable=pose_var,
        values=("mediapipe", "trt"),
        width=14,
        state="readonly",
    ).grid(row=0, column=1, sticky="w", padx=10, pady=6)
    ttk.Label(runtime_frame, text="Camera").grid(row=1, column=0, sticky="w", padx=10, pady=6)
    ttk.Combobox(
        runtime_frame,
        textvariable=camera_var,
        values=("realsense", "webcam"),
        width=14,
        state="readonly",
    ).grid(row=1, column=1, sticky="w", padx=10, pady=6)

    ttk.Label(runtime_frame, text="Capture").grid(row=2, column=0, sticky="w", padx=10, pady=6)
    ttk.Label(runtime_frame, text="640 x 480 @ 30 FPS").grid(
        row=2, column=1, sticky="w", padx=10, pady=6
    )

    model_options = ttk.LabelFrame(main, text="Model Options")
    model_options.grid(row=3, column=0, sticky="ew", pady=(0, 10))
    model_options.columnconfigure(1, weight=1)
    ttk.Label(model_options, text="Intersection fold").grid(row=0, column=0, sticky="w", padx=10, pady=6)
    ttk.Spinbox(model_options, from_=1, to=5, textvariable=fold_var, width=6).grid(
        row=0, column=1, sticky="w", padx=10, pady=6
    )
    ttk.Label(model_options, text="Device").grid(row=1, column=0, sticky="w", padx=10, pady=6)
    ttk.Entry(model_options, textvariable=device_var, width=18).grid(
        row=1, column=1, sticky="w", padx=10, pady=6
    )
    ttk.Checkbutton(
        model_options,
        text="Swap bus upper-limb left/right display labels",
        variable=swap_upper_var,
    ).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=6)

    recording = ttk.LabelFrame(main, text="Recording")
    recording.grid(row=4, column=0, sticky="ew", pady=(0, 12))
    recording.columnconfigure(1, weight=1)
    ttk.Checkbutton(
        recording,
        text="Headless: save annotated video instead of opening live window",
        variable=headless_var,
    ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=6)
    ttk.Label(recording, text="Duration seconds").grid(row=1, column=0, sticky="w", padx=10, pady=6)
    ttk.Entry(recording, textvariable=duration_var, width=12).grid(
        row=1, column=1, sticky="w", padx=10, pady=6
    )
    ttk.Label(recording, text="Output file").grid(row=2, column=0, sticky="w", padx=10, pady=6)
    ttk.Entry(recording, textvariable=output_var).grid(row=2, column=1, sticky="ew", padx=10, pady=6)

    status_var = tk.StringVar(value="")
    ttk.Label(main, textvariable=status_var, foreground="#b00020").grid(
        row=5, column=0, sticky="w", pady=(0, 8)
    )

    buttons = ttk.Frame(main)
    buttons.grid(row=6, column=0, sticky="e")

    def start():
        try:
            duration_text = duration_var.get().strip()
            duration = float(duration_text) if duration_text else None
            if duration is not None and duration <= 0:
                raise ValueError("Duration must be positive.")
        except ValueError as exc:
            status_var.set(str(exc))
            return

        updated = copy.copy(args)
        updated.model = model_var.get()
        updated.pose = pose_var.get()
        updated.camera = camera_var.get()
        updated.width = 640
        updated.height = 480
        updated.fps = 30
        updated.headless = bool(headless_var.get())
        updated.duration = duration
        updated.output = output_var.get().strip() or None
        updated.bus_swap_upper_labels = bool(swap_upper_var.get())
        updated.intersection_fold = int(fold_var.get())
        updated.device = device_var.get().strip() or None
        selected["args"] = updated
        root.destroy()

    def cancel():
        selected["args"] = None
        root.destroy()

    ttk.Button(buttons, text="Cancel", command=cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Start", command=start).grid(row=0, column=1)

    root.mainloop()
    return selected["args"]


def choose_options(args):
    try:
        return choose_options_gui(args)
    except Exception as exc:
        print(f"GUI unavailable ({exc}); falling back to terminal menu.")
        updated = copy.copy(args)
        updated.model = choose_model_cli()
        return updated


def fill_runtime_defaults(args):
    if args.headless and not args.output:
        args.output = (
            "bus_stop_output.avi"
            if args.model == "bus"
            else "intersection_output.avi"
        )
    return args


def main():
    args = parse_args()
    if args.model is None:
        args = choose_options(args)
        if args is None:
            return 0

    args = fill_runtime_defaults(args)
    predictor = create_predictor(args)
    run_realtime(args, predictor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
