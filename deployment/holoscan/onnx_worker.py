"""Process-isolated ONNX inference for Holoscan Python operators.

GXF invokes Python operators from a native worker thread. Some native Python
inference extensions can crash while releasing their Python thread state from
that context. Running ONNX Runtime in a spawned process keeps inference on the
child process's main Python thread and leaves the GXF worker with only standard
library IPC.
"""

import multiprocessing
import queue
import traceback
from types import SimpleNamespace


def _worker_main(args_dict, requests, responses):
    try:
        from unified_prediction.predictors import create_predictor

        predictor = create_predictor(SimpleNamespace(**args_dict))
        responses.put(
            {
                "kind": "ready",
                "title": predictor.title,
                "lines": predictor.overlay_lines(),
                "progress": predictor.progress(),
            }
        )
        while True:
            request = requests.get()
            if request is None:
                break
            request_id, keypoints, timestamp = request
            predictor.update(keypoints, timestamp)
            responses.put(
                {
                    "kind": "result",
                    "request_id": request_id,
                    "lines": predictor.overlay_lines(),
                    "progress": predictor.progress(),
                }
            )
    except BaseException:
        responses.put({"kind": "error", "traceback": traceback.format_exc()})


class OnnxInferenceProcess:
    """Predictor-compatible proxy backed by a spawned ONNX worker process."""

    def __init__(self, args):
        self._timeout = float(getattr(args, "onnx_worker_timeout", 120.0))
        self._context = multiprocessing.get_context("spawn")
        self._requests = self._context.Queue(maxsize=1)
        self._responses = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_worker_main,
            args=(vars(args).copy(), self._requests, self._responses),
            name="holoscan-onnx-inference",
            daemon=True,
        )
        self._request_id = 0
        self._lines = []
        self._progress = (0, 1)
        self.title = "Bus-stop model: straight / yield / overtake"
        self._process.start()
        message = self._receive("ONNX worker startup")
        if message.get("kind") != "ready":
            self.close()
            raise RuntimeError(f"Unexpected ONNX worker startup response: {message!r}")
        self.title = message["title"]
        self._lines = message["lines"]
        self._progress = tuple(message["progress"])
        print(
            f"Started isolated ONNX inference process pid={self._process.pid}",
            flush=True,
        )

    def _receive(self, operation):
        try:
            message = self._responses.get(timeout=self._timeout)
        except queue.Empty as exc:
            exitcode = self._process.exitcode
            raise RuntimeError(
                f"{operation} timed out after {self._timeout:g}s "
                f"(worker exitcode={exitcode})"
            ) from exc
        if message.get("kind") == "error":
            raise RuntimeError(
                f"{operation} failed in the ONNX worker:\n{message['traceback']}"
            )
        return message

    def update(self, keypoints, timestamp):
        if not self._process.is_alive():
            raise RuntimeError(
                f"ONNX worker exited unexpectedly with code {self._process.exitcode}"
            )
        request_id = self._request_id
        self._request_id += 1
        self._requests.put((request_id, keypoints, timestamp), timeout=self._timeout)
        message = self._receive(f"ONNX inference request {request_id}")
        if message.get("kind") != "result" or message.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected ONNX worker response: {message!r}")
        self._lines = message["lines"]
        self._progress = tuple(message["progress"])

    def overlay_lines(self):
        return self._lines

    def progress(self):
        return self._progress

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.is_alive():
            try:
                self._requests.put(None, timeout=1.0)
            except queue.Full:
                pass
            process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        self._requests.close()
        self._responses.close()
        self._process = None
