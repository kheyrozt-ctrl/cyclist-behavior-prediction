# Cyclist Behavior Prediction

Academic submission and open-source release repository for cyclist/VRU behavior
prediction, public dataset preparation, and CAVE-to-cloud-to-edge deployment.

## Main Deliverables

### Public anonymized VRU dataset

Location: `public_dataset/`

- Dataset release documentation and inventory.
- Hugging Face dataset preparation scripts.
- Privacy checking and public training notes.

### Open-source training/inference code

Location: `open_source/`

- `training/`: model training and evaluation code.
- `training/intersection_intention_legacy/`: original intersection gesture-to-maneuver training lineage.
- `training/bus_stop_v5_public/`: V5 public-training and combined bus-stop model lineage.
- `inference/intersection_runtime/`: original real-time intersection inference runtime.
- `inference/unified_prediction/`: unified prediction runtime and web launcher.
- `model_artifacts/bus_stop_torchscript/`: packaged bus-stop TorchScript artifacts.
- `pose_backends/trt_pose/`: Jetson/TRT pose backend support.
- `launchers/`: launcher scripts and dependency files.

License: Apache-2.0, as declared by the repository `LICENSE`.

### Holoscan edge graphs

Location: `edge_deployment/`

- Holoscan app, config, simulation, and validation notes.

### CAVE-to-cloud-to-edge framework

Location: `framework_blueprint/`

- Framework blueprint and best-practice guide.

## Start Prediction

Prerequisite: install 64-bit Python 3.11 or 3.12 on desktop systems. Jetson
systems may use JetPack's aarch64 Python 3.8 environment. Python 3.13 is not
supported by the current NumPy/MediaPipe dependency set.

Recommended Windows entrypoint:

```powershell
.\start_prediction.bat
```

Direct runtime entrypoint:

```powershell
cd open_source\inference\unified_prediction
.\start_web.bat
```

The unified runtime creates its own `.venv` in
`open_source/inference/unified_prediction/` and installs
`open_source/inference/unified_prediction/requirements.txt`.

## Important Model Note

The bus-stop unified runtime includes packaged TorchScript models. The
intersection pipeline still requires explicit and implicit gesture checkpoints;
see `open_source/inference/unified_prediction/README.md` for expected paths.

## Repository Map

See `MANIFEST.md` for the full submission-oriented file map.
