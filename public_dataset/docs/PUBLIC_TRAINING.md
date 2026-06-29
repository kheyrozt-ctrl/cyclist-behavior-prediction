# Public Training Workflow

## Canonical public input

Use only the anonymized dataset
[`Kheyro/cyclist-intention-2d-keypoints`](https://huggingface.co/datasets/Kheyro/cyclist-intention-2d-keypoints)
for public reproduction. It is licensed under CC BY 4.0 and contains
release-scoped participant and sequence identifiers. Original filenames,
recording dates, session identifiers, source paths, raw video, and the private
identifier mapping are not required.

The legacy `VideoCrop` builders remain in this repository to document the local
source-processing lineage. Their generated CSV/TXT manifests are intentionally
excluded from Git because they contain source-system identifiers.

## Prepare Stage2 inputs

Install release dependencies:

```bash
python -m pip install -r requirements-release.txt
```

Install PyTorch using the package appropriate for the target CPU, CUDA, or
Jetson environment, then prepare one participant-disjoint fold:

```bash
python tools/prepare_hf_stage2.py --fold 1
```

The command downloads the three Parquet splits and
`participant_folds.parquet`, verifies the expected schema, keeps only
Stage2-eligible sequences, resamples each sequence at 12 Hz, and writes the
`fold_1_{train,val,test}_data.pkl` files consumed by:

```bash
python open_source/training/bus_stop_v5_public/Gesture2Manuever_Prediction/ManueverPrediction_Combined/main.py \
  --fold 1 \
  --data-dir release_data/ManueverDataset \
  --stage1-checkpoint-dir PATH_TO_STAGE1_CHECKPOINTS
```

Use `--fold 1` through `--fold 5` to reproduce another published participant
fold. Generated PKL files and checkpoints are local artifacts and must not be
committed.

## Reproducibility boundary

The public dataset is sufficient to reconstruct Stage2 raw skeleton windows.
The V5 combined model also requires matching Stage1 branch checkpoints. Train
those from the published gesture annotations or provide explicitly versioned
checkpoints; do not silently substitute legacy checkpoints trained from a
different dataset revision.

No accuracy, latency, throughput, power, or thermal result is asserted by this
workflow. Publish such results only with the code commit, dataset revision,
fold, seed, complete configuration, hardware/software profile, and evaluation
artifact.

Before publishing a repository revision, run:

```bash
python tools/check_public_privacy.py
```
