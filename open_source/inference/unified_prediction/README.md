# Unified Prediction Package

This directory contains the local runtime code and bundled artifacts used by
the unified cyclist prediction launcher.

## Launch

Use 64-bit Python 3.11 or 3.12. Python 3.13 is not supported by the current
NumPy/MediaPipe dependency set.

Windows:

```bat
start_web.bat
```

Linux/macOS:

```sh
./start_web.sh
```

The scripts create a `.venv` inside this directory and install
`requirements.txt`.

## Bundled Local Artifacts

- `model_package/`: bus-stop TorchScript models and config.
- `models.py`: LSTM definitions for the intersection pipeline.
- `pose_adapter.py`: MediaPipe and TRT pose adapter.
- `pose_detection/`: local TRT pose support files copied from the root
  `pose_detection` directory.
- `checkpoints/intersection/ManueverPrediction/Models/`: available maneuver
  checkpoints.

## Missing Intersection Checkpoints

The current repository does not include the default explicit and implicit
gesture checkpoints expected by the intersection pipeline:

- `checkpoints/intersection/ExplicitModels/best_model_fold_*.pkl`
- `checkpoints/intersection/ImplicitModels/best_model_fold_*.pkl`

Place those files in the paths above, or pass explicit checkpoint paths with
`--intersection-explicit-model` and `--intersection-implicit-model`.
