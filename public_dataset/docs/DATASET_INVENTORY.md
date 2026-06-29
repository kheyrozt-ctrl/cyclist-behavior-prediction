# Local Dataset Inventory

> This is a historical inventory of an ignored local working copy, not an
> inventory of files published on GitHub or Hugging Face. The current public
> training source is the anonymized V5 Hugging Face release described in the
> final section. Original filenames and local label/split manifests are not
> part of the public Git tree.

## Scope

This inventory describes the dataset material currently visible in the local
workspace. These paths are excluded by `.gitignore`; they are not part of the
source repository and must not be added to Git without a separate data-release,
license, privacy, and size review.

Inventory date: 2026-06-23.

## Observed data

| Path | Files | Approximate size | Observed formats | Role |
|---|---:|---:|---|---|
| `open_source/training/intersection_intention_legacy/Full33SkeletonData/` | 226 | 256.86 MB | 226 CSV | MediaPipe-style 33-joint skeleton sequences |
| `open_source/training/intersection_intention_legacy/GestureDataset/` | 618 | 3,011.24 MB | 392 CSV, 218 MP4, 5 PKL, 2 TXT, 1 XLSX | Raw/derived trial video, parsed telemetry, survey and label material |
| `open_source/training/intersection_intention_legacy/Results/` | 41 | 3.21 MB | 21 NPY, 20 CSV | Derived labels, masks, and trajectory outputs |

Counts describe this working copy and can change independently of Git.

## Skeleton schema

The sampled files in `Full33SkeletonData/` use 67 columns:

```text
timestamp,
NOSE_x, NOSE_y,
LEFT_EYE_INNER_x, LEFT_EYE_INNER_y,
...
```

This is one timestamp plus 33 two-dimensional keypoints (`33 × 2 = 66`), which
matches the feature width expected by the unified predictors. Filename patterns
such as `video1a_10_<id>_<date>_<time>_skeleton.csv` retain scenario/session
identifiers and must be handled as potentially identifying research metadata.

The source paper reports 31 study participants, trial metadata lists 30 participant
IDs, and the available skeleton subset resolves to 29 participants. Participant
14 has two source sessions; filename tokens ending in `141` and `142` are session
variants, not separate participants. Release splits must use the trial metadata
mapping rather than parsing participant identity directly from filename tokens.

## Gesture dataset layout

Observed top-level entries include:

```text
GestureDataset/
├── depth_camera_videos_croped/
├── parsed_data/
├── result_data/
├── survey_data/
├── Videos_croped_withprobleminexp/
├── ManueverClass.txt
└── Simulator Trial Notes.csv
```

`Simulator Trial Notes.csv` contains scenario, participant, source-file, usage,
discard-code, head-yaw, and notes fields. Parsed trial CSV files are heterogeneous:
sampled files range from 110 to 159 columns and include time, simulator time,
helmet/head yaw, coordinate, and score fields. Code consuming these files must
select columns by name and version rather than assuming a fixed positional schema.

## Results layout

The current `Results/` directory contains `LabeledTrajDataset/`, `Masks/`, and
`mask_data.npy`. Treat these as derived artifacts: their reproducibility depends
on the exact source files, labeling code, configuration, and split definition.

## Use with the unified runtime

The web launcher performs live inference and does not load these dataset folders.
Its model inputs are live 33-keypoint frames produced by the selected pose adapter.
The local datasets are relevant to offline validation, replay, retraining, and
dataset-quality work, not to basic launcher startup.

Before using local files for training or evaluation:

1. define the exact task and label source;
2. create participant-disjoint train, validation, and test manifests;
3. record excluded/discarded trials and reasons;
4. validate required columns, timestamp monotonicity, missing values, and frame rate;
5. hash source files and store the transformation/code revision;
6. confirm consent, license, and permitted-use constraints;
7. keep raw video, survey data, and participant mappings out of Git.

The broader lineage, privacy, export, and split requirements are defined in
`docs/CAVE_CLOUD_EDGE_FRAMEWORK_BLUEPRINT.md`.

## Public release status

The Hugging Face repository was reset on 2026-06-24 using
`open_source/training/bus_stop_v5_public` as the sole dataset source. The current release is
[Kheyro/cyclist-intention-2d-keypoints](https://huggingface.co/datasets/Kheyro/cyclist-intention-2d-keypoints)
under CC BY 4.0. V2 contains 32 participants, 559 scene clips, 99,734 frames,
2,086 sanitized Stage1 annotation rows, 357 Stage2-eligible clips, and all five
participant-level folds. The previous intersection release is superseded and no
longer present in the remote repository. See `docs/PUBLIC_DATASET_RELEASE.md`.
