# Public Dataset Release

## Current release

| Field | Value |
|---|---|
| Repository | [Kheyro/cyclist-intention-2d-keypoints](https://huggingface.co/datasets/Kheyro/cyclist-intention-2d-keypoints) |
| Version | 2.0.0 |
| Release date | 2026-06-24 |
| Hugging Face commit | `072d681aee1e5569c64572bc1546b5b5364f4413` |
| License | CC BY 4.0 |
| Canonical source | `0cyclistprediction_v5/` |
| Participants | 32 |
| Scene clips | 559 |
| Frames | 99,734 |

Version 2.0.0 is a breaking reset of the Hugging Face repository. The earlier
intersection dataset was removed from the current repository contents. V2 is the
V5 bus-stop/road cyclist-behavior dataset with Stage2 labels `straight`, `yield`,
and `overtake`.

## Published files

```text
README.md
LICENSE.md
dataset_manifest.json
privacy_report.json
release_stats.json
sequences.csv
annotations.parquet
participant_folds.parquet
data/
├── train-00000-of-00001.parquet
├── validation-00000-of-00001.parquet
└── test-00000-of-00001.parquet
```

The three `data/` files contain frame-level 33×2D pose coordinates, relative
time, release-scoped sequence/participant IDs, fixed FPS, route anchor, maneuver
label, and Stage2 eligibility. `annotations.parquet` contains sanitized Stage1
gesture intervals and blocker status. `participant_folds.parquet` publishes all
five participant-level folds.

## Canonical Fold 1

| Split | Participants | Clips | Frames |
|---|---:|---:|---:|
| Train | 21 | 369 | 65,235 |
| Validation | 6 | 103 | 18,491 |
| Test | 5 | 87 | 16,008 |

All five supplied folds were validated as participant-disjoint.

## Labels and training relationship

Stage1 annotations cover head direction, upper-limb rotation, and pedaling. The
release contains 2,086 sanitized annotation rows; excluded/disabled intervals
remain marked so consumers can reproduce blocker logic.

Stage2 predicts `straight`, `yield`, and `overtake`. Of the 559 public scene
clips, 357 meet the supplied Stage2 eligibility rules. Scene clips at excluded
route anchors or listed whole-clip blockers remain useful for Stage1 and are
published with `stage2_eligible=false`.

The corresponding four-step training implementation is under
`0cyclistprediction_v5/Gesture2Manuever_Prediction/`.

## Privacy transformation

The public release removes raw video, original filenames, calendar dates,
session IDs, direct participant numbers, and source filesystem paths. Original
timestamps are converted to time relative to the first frame of each clip.

Participant and sequence IDs are release-scoped HMAC values. The private key is
stored under ignored `.release-secrets/` and is not uploaded. Pose trajectories
can still retain behavioral signatures; re-identification and external identity
linkage are prohibited in the Dataset Card.

## Reproduce

```powershell
.venv\Scripts\python -m pip install -r requirements-release.txt
.venv\Scripts\python tools\build_hf_dataset_v5.py `
  --license-approved-by "Project owner confirmation, 2026-06-24" `
  --repo-id "Kheyro/cyclist-intention-2d-keypoints"
```

The generated package is written to
`release/hf/cyclist-intention-2d-keypoints-v2/`. Generated release data and
pseudonymization secrets are excluded from Git.

## Validation performed

- exactly 559 skeleton CSV files with timestamp plus 66 pose columns;
- 32 participants represented in all five supplied fold definitions;
- no participant overlap between train, validation, and test in any fold;
- every clip resolves to consistent FPS, route anchor, and maneuver metadata;
- public Parquet schemas contain no raw identifier columns;
- text artifacts contain no original participant/session filename patterns;
- remote repository contains no legacy V1 files;
- public Dataset Viewer successfully loads V2 frame rows and maneuver labels.
