"""Prediction backends used by the unified runtime."""

import collections
import json
import os
import sys

import numpy as np
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CYCLIST_INFERENCE_DIR = os.path.join(ROOT_DIR, "cyclist_inference")
if CYCLIST_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, CYCLIST_INFERENCE_DIR)

from models import (  # noqa: E402
    ExplicitLSTMClassifier,
    ImplicitLSTMClassifier,
    ManeuverLSTMClassifier,
)


def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def softmax_label(logits, labels):
    probs = torch.softmax(logits, dim=1)
    idx = int(torch.argmax(probs, dim=1).item())
    return labels[idx], float(probs[0, idx].item()), idx


def numpy_softmax_label(logits, labels):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    idx = int(np.argmax(probabilities, axis=1)[0])
    return labels[idx], float(probabilities[0, idx]), idx


def format_prediction(prefix, pred):
    if pred is None:
        return f"{prefix}: collecting..."
    label, prob, _ = pred
    return f"{prefix}: {label} ({prob:.2f})"


class RateSampler:
    def __init__(self, fps):
        self.interval = 1.0 / fps
        self.last_sample_time = None

    def should_sample(self, now):
        if self.last_sample_time is None:
            self.last_sample_time = now
            return True
        if now - self.last_sample_time >= self.interval:
            self.last_sample_time = now
            return True
        return False


class BusStopPredictor:
    title = "Bus-stop model: straight / yield / overtake"

    HEAD_WINDOW = 12
    UPPER_WINDOW = 12
    LEG_WINDOW = 20
    STAGE2_SEQ_LEN = 120
    STAGE1_FPS = 12.0
    LEG_FPS = 20.0

    def __init__(self, args):
        package_dir = os.path.join(ROOT_DIR, "model_package")
        config_path = args.bus_config or os.path.join(package_dir, "model_config.json")
        require_file(config_path, "Bus config")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.labels = {
            "head": config["stage1"]["head"]["classes"],
            "upper_limb": config["stage1"]["upper_limb"]["classes"],
            "leg": config["stage1"]["leg"]["classes"],
            "stage2": config["stage2"]["classes"],
        }
        if args.bus_swap_upper_labels:
            self.labels["upper_limb"] = self.labels["upper_limb"].copy()
            self.labels["upper_limb"][0], self.labels["upper_limb"][1] = (
                self.labels["upper_limb"][1],
                self.labels["upper_limb"][0],
            )
            print("Swapped upper-limb left/right display labels.")

        if args.device and args.device != "cpu":
            print(
                "WARNING: Bus TorchScript models create LSTM hidden states on CPU. "
                "Forcing bus model inference to CPU."
            )
        self.device = torch.device("cpu")

        self.models = {
            "head": self._load_torchscript(
                args.bus_head_model or os.path.join(package_dir, "head_model.pt"),
                "Head model",
            ),
            "upper": self._load_torchscript(
                args.bus_upper_model or os.path.join(package_dir, "upper_model.pt"),
                "Upper-limb model",
            ),
            "leg": self._load_torchscript(
                args.bus_leg_model or os.path.join(package_dir, "leg_model.pt"),
                "Leg model",
            ),
            "stage2": self._load_torchscript(
                args.bus_stage2_model or os.path.join(package_dir, "stage2_maneuver_model.pt"),
                "Stage2 model",
            ),
        }
        print(f"Loaded bus-stop TorchScript models on {self.device}")

        self.buffers = {
            "head": collections.deque(maxlen=self.HEAD_WINDOW),
            "upper": collections.deque(maxlen=self.UPPER_WINDOW),
            "leg": collections.deque(maxlen=self.LEG_WINDOW),
        }
        self.feature_buffer = collections.deque(maxlen=self.STAGE2_SEQ_LEN)
        self.stage1_sampler = RateSampler(self.STAGE1_FPS)
        self.leg_sampler = RateSampler(self.LEG_FPS)
        self.stage1_pred = {}
        self.stage2_pred = None

    def _load_torchscript(self, path, label):
        require_file(path, label)
        model = torch.jit.load(path, map_location=self.device)
        model.to(self.device).eval()
        return model

    def update(self, keypoints, now):
        if self.leg_sampler.should_sample(now):
            self.buffers["leg"].append(keypoints)
        if not self.stage1_sampler.should_sample(now):
            return

        self.buffers["head"].append(keypoints)
        self.buffers["upper"].append(keypoints)

        ready = (
            len(self.buffers["head"]) == self.HEAD_WINDOW
            and len(self.buffers["upper"]) == self.UPPER_WINDOW
            and len(self.buffers["leg"]) == self.LEG_WINDOW
        )
        if not ready:
            return

        feature, self.stage1_pred = self._run_stage1()
        self.feature_buffer.append(feature)
        if len(self.feature_buffer) == self.STAGE2_SEQ_LEN:
            self.stage2_pred = self._run_stage2()

    def _run_stage1(self):
        head_x = torch.tensor(np.array(self.buffers["head"], dtype=np.float32),
                              device=self.device).unsqueeze(0)
        upper_x = torch.tensor(np.array(self.buffers["upper"], dtype=np.float32),
                               device=self.device).unsqueeze(0)
        leg_x = torch.tensor(np.array(self.buffers["leg"], dtype=np.float32),
                             device=self.device).unsqueeze(0)

        with torch.no_grad():
            head_logits, head_feature = self.models["head"](head_x)
            upper_logits, upper_feature = self.models["upper"](upper_x)
            leg_logits, leg_feature = self.models["leg"](leg_x)

        feature = torch.cat([head_feature, upper_feature, leg_feature], dim=1)
        if feature.shape[1] != 640:
            raise RuntimeError(f"Stage1 feature dim should be 640, got {feature.shape[1]}")

        predictions = {
            "head": softmax_label(head_logits, self.labels["head"]),
            "upper": softmax_label(upper_logits, self.labels["upper_limb"]),
            "leg": softmax_label(leg_logits, self.labels["leg"]),
        }
        return feature.squeeze(0).detach().cpu().numpy(), predictions

    def _run_stage2(self):
        seq = torch.tensor(np.array(self.feature_buffer, dtype=np.float32),
                           device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.models["stage2"](seq)
        return softmax_label(logits, self.labels["stage2"])

    def overlay_lines(self):
        return [
            (format_prediction("Head", self.stage1_pred.get("head")), (210, 255, 210)),
            (format_prediction("Upper", self.stage1_pred.get("upper")), (210, 255, 210)),
            (format_prediction("Leg", self.stage1_pred.get("leg")), (210, 255, 210)),
            (format_prediction("Maneuver", self.stage2_pred), (80, 220, 255)),
            (f"Stage2 feature buffer: {len(self.feature_buffer)}/{self.STAGE2_SEQ_LEN}",
             (255, 255, 255)),
        ]

    def progress(self):
        return len(self.feature_buffer), self.STAGE2_SEQ_LEN


class OnnxBusStopPredictor(BusStopPredictor):
    """Bus-stop predictor using ONNX Runtime instead of Python PyTorch."""

    def __init__(self, args):
        import onnxruntime as ort

        package_dir = os.path.join(ROOT_DIR, "model_package")
        config_path = args.bus_config or os.path.join(package_dir, "model_config.json")
        require_file(config_path, "Bus config")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.labels = {
            "head": config["stage1"]["head"]["classes"],
            "upper_limb": config["stage1"]["upper_limb"]["classes"],
            "leg": config["stage1"]["leg"]["classes"],
            "stage2": config["stage2"]["classes"],
        }
        if args.bus_swap_upper_labels:
            self.labels["upper_limb"] = self.labels["upper_limb"].copy()
            self.labels["upper_limb"][0], self.labels["upper_limb"][1] = (
                self.labels["upper_limb"][1],
                self.labels["upper_limb"][0],
            )

        model_dir = getattr(args, "onnx_model_dir", None)
        if model_dir is None:
            raise ValueError("--onnx-model-dir is required with --runtime onnx")
        model_dir = os.fspath(model_dir)
        paths = {
            "head": os.path.join(model_dir, "head.onnx"),
            "upper": os.path.join(model_dir, "upper.onnx"),
            "leg": os.path.join(model_dir, "leg.onnx"),
            "stage2": os.path.join(model_dir, "stage2.onnx"),
        }
        for key, path in paths.items():
            require_file(path, f"{key} ONNX model")

        provider = getattr(args, "onnx_provider", "CPUExecutionProvider")
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"ONNX provider {provider!r} is unavailable; available={available}"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(getattr(args, "onnx_threads", 1)))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sessions = {
            key: ort.InferenceSession(
                path,
                sess_options=options,
                providers=[provider],
            )
            for key, path in paths.items()
        }
        self.buffers = {
            "head": collections.deque(maxlen=self.HEAD_WINDOW),
            "upper": collections.deque(maxlen=self.UPPER_WINDOW),
            "leg": collections.deque(maxlen=self.LEG_WINDOW),
        }
        self.feature_buffer = collections.deque(maxlen=self.STAGE2_SEQ_LEN)
        self.stage1_sampler = RateSampler(self.STAGE1_FPS)
        self.leg_sampler = RateSampler(self.LEG_FPS)
        self.stage1_pred = {}
        self.stage2_pred = None
        print(f"Loaded bus-stop ONNX models with {provider}")

    def _run_stage1(self):
        inputs = {
            "head": np.asarray(self.buffers["head"], dtype=np.float32)[None, ...],
            "upper": np.asarray(self.buffers["upper"], dtype=np.float32)[None, ...],
            "leg": np.asarray(self.buffers["leg"], dtype=np.float32)[None, ...],
        }
        outputs = {
            key: self.sessions[key].run(None, {"input": values})
            for key, values in inputs.items()
        }
        feature = np.concatenate(
            [outputs["head"][1], outputs["upper"][1], outputs["leg"][1]],
            axis=1,
        )
        if feature.shape != (1, 640):
            raise RuntimeError(f"Stage1 feature shape should be (1, 640), got {feature.shape}")
        predictions = {
            "head": numpy_softmax_label(outputs["head"][0], self.labels["head"]),
            "upper": numpy_softmax_label(
                outputs["upper"][0],
                self.labels["upper_limb"],
            ),
            "leg": numpy_softmax_label(outputs["leg"][0], self.labels["leg"]),
        }
        return feature[0], predictions

    def _run_stage2(self):
        sequence = np.asarray(self.feature_buffer, dtype=np.float32)[None, ...]
        logits = self.sessions["stage2"].run(None, {"input": sequence})[0]
        return numpy_softmax_label(logits, self.labels["stage2"])


class IntersectionPredictor:
    title = "Intersection model: Crossing / LeftTurn / RightTurn"

    EXPLICIT_LABELS = ["Right_Hand_Explicit", "neutral_explicit", "Left_Hand_Explicit"]
    EXPLICIT_WEIGHTS = {
        "Right_Hand_Explicit": 1,
        "neutral_explicit": 0,
        "Left_Hand_Explicit": -1,
    }
    IMPLICIT_LABELS = ["Right_Look", "neutral_implicit", "Left_Overshoulder", "Left_Look"]
    IMPLICIT_WEIGHTS = {
        "Right_Look": 1,
        "neutral_implicit": 0,
        "Left_Overshoulder": -2,
        "Left_Look": -1,
    }
    MANEUVER_LABELS = ["Crossing", "LeftTurn", "RightTurn"]
    WINDOW_SIZE = 30
    GESTURE_SEQ_LEN = 719

    def __init__(self, args):
        if args.device:
            self.device = torch.device(args.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print(f"Using intersection device: {self.device}")

        explicit_path, implicit_path, maneuver_path = self._resolve_model_paths(args)
        self.explicit_model = ExplicitLSTMClassifier()
        self.implicit_model = ImplicitLSTMClassifier()
        self.maneuver_model = ManeuverLSTMClassifier()

        self.explicit_model.load_state_dict(torch.load(explicit_path, map_location=self.device))
        self.implicit_model.load_state_dict(torch.load(implicit_path, map_location=self.device))
        self.maneuver_model.load_state_dict(torch.load(maneuver_path, map_location=self.device))

        self.explicit_model.to(self.device).eval()
        self.implicit_model.to(self.device).eval()
        self.maneuver_model.to(self.device).eval()
        print(f"Loaded intersection models on {self.device}")

        self.keypoint_buffer = collections.deque(maxlen=self.WINDOW_SIZE)
        self.gesture_buffer = collections.deque(maxlen=self.GESTURE_SEQ_LEN)
        self.explicit_label = None
        self.implicit_label = None
        self.maneuver_label = None

    def _resolve_model_paths(self, args):
        base_dir = os.path.join(
            ROOT_DIR,
            "cyclistprediction",
            "Gesture2Manuever_Prediction",
        )
        explicit_default = os.path.join(
            base_dir,
            "GestureClassification",
            "ExplicitModels",
            "best_model_fold_{}.pkl",
        )
        implicit_default = os.path.join(
            base_dir,
            "GestureClassification",
            "ImplicitModels",
            "best_model_fold_{}.pkl",
        )
        maneuver_default = os.path.join(
            base_dir,
            "ManueverPrediction",
            "Models",
            "best_model_fold_{}.pkl",
        )

        explicit = args.intersection_explicit_model or explicit_default.format(args.intersection_fold)
        implicit = args.intersection_implicit_model or implicit_default.format(args.intersection_fold)
        maneuver = args.intersection_maneuver_model or maneuver_default.format(args.intersection_fold)

        for label, path in [
            ("Explicit model", explicit),
            ("Implicit model", implicit),
            ("Maneuver model", maneuver),
        ]:
            require_file(path, label)
        return explicit, implicit, maneuver

    def update(self, keypoints, now):
        del now
        self.keypoint_buffer.append(keypoints)
        if len(self.keypoint_buffer) != self.WINDOW_SIZE:
            return

        ex_idx = self._classify_gesture(self.explicit_model)
        im_idx = self._classify_gesture(self.implicit_model)
        self.explicit_label = self.EXPLICIT_LABELS[ex_idx]
        self.implicit_label = self.IMPLICIT_LABELS[im_idx]

        ex_weight = self.EXPLICIT_WEIGHTS[self.explicit_label]
        im_weight = self.IMPLICIT_WEIGHTS[self.implicit_label]
        self.gesture_buffer.append([ex_weight, im_weight])

        if len(self.gesture_buffer) == self.GESTURE_SEQ_LEN:
            idx = self._predict_maneuver()
            self.maneuver_label = self.MANEUVER_LABELS[idx]

    def _classify_gesture(self, model):
        window = np.array(self.keypoint_buffer, dtype=np.float32)
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(tensor)
        return int(torch.argmax(output, dim=1).item())

    def _predict_maneuver(self):
        seq = np.array(self.gesture_buffer, dtype=np.float32)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.maneuver_model(tensor)
        return int(torch.argmax(output, dim=1).item())

    def overlay_lines(self):
        return [
            (f"Explicit: {self.explicit_label or 'waiting...'}", (210, 255, 210)),
            (f"Implicit: {self.implicit_label or 'waiting...'}", (210, 255, 210)),
            (f"Maneuver: {self.maneuver_label or 'collecting...'}", (80, 220, 255)),
            (f"Gesture buffer: {len(self.gesture_buffer)}/{self.GESTURE_SEQ_LEN}",
             (255, 255, 255)),
        ]

    def progress(self):
        return len(self.gesture_buffer), self.GESTURE_SEQ_LEN


def create_predictor(args):
    if args.model == "bus":
        if getattr(args, "runtime", "torch") == "onnx":
            return OnnxBusStopPredictor(args)
        return BusStopPredictor(args)
    if args.model == "intersection":
        return IntersectionPredictor(args)
    raise ValueError(f"Unknown model: {args.model}")
