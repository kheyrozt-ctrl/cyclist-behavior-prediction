from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


REPO_ID = "Kheyro/cyclist-intention-2d-keypoints"
DATA_FILES = (
    "data/train-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
)
FOLDS_FILE = "participant_folds.parquet"
LABELS = ("straight", "yield", "overtake")
STEP_FRAMES = {"straight": 2, "yield": 1, "overtake": 1}
TARGET_FPS = 12.0
WINDOW_FRAMES = 120
METADATA_COLUMNS = {
    "sequence_id",
    "participant_id",
    "frame_index",
    "relative_time_s",
    "fixed_fps",
    "route_anchor",
    "maneuver_label",
    "stage2_eligible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare anonymized Hugging Face cyclist data for V5 Stage2 training."
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release_data") / "ManueverDataset",
    )
    parser.add_argument(
        "--local-dataset-dir",
        type=Path,
        help="Use an existing dataset snapshot instead of downloading it.",
    )
    return parser.parse_args()


def resolve_file(args: argparse.Namespace, filename: str) -> Path:
    if args.local_dataset_dir is not None:
        path = args.local_dataset_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            revision=args.revision,
        )
    )


def load_fold_membership(path: Path, fold: int) -> dict[str, str]:
    table = pq.read_table(path, columns=["fold", "subset", "participant_id"])
    membership: dict[str, str] = {}
    for row in table.to_pylist():
        if int(row["fold"]) != fold:
            continue
        subset = {"validation": "val"}.get(str(row["subset"]), str(row["subset"]))
        if subset not in {"train", "val", "test"}:
            raise ValueError(f"unsupported subset: {subset}")
        participant_id = str(row["participant_id"])
        if participant_id in membership:
            raise ValueError(f"duplicate fold assignment: {participant_id}")
        membership[participant_id] = subset
    if not membership:
        raise ValueError(f"fold {fold} has no participant assignments")
    return membership


def load_frames(paths: list[Path]) -> pa.Table:
    tables = [pq.read_table(path) for path in paths]
    schema_names = set(tables[0].schema.names)
    required = METADATA_COLUMNS | {"nose_x", "right_foot_index_y"}
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    for table in tables[1:]:
        if table.schema != tables[0].schema:
            raise ValueError("Parquet split schemas do not match")
    return pa.concat_tables(tables)


def fill_and_resample(times: np.ndarray, coords: np.ndarray) -> np.ndarray:
    order = np.argsort(times)
    times = times[order]
    coords = coords[order]
    times, unique_indices = np.unique(times, return_index=True)
    coords = coords[unique_indices]
    if len(times) < 2:
        return coords.astype("float32")

    for column in range(coords.shape[1]):
        values = coords[:, column]
        valid = np.isfinite(values)
        if valid.any():
            coords[:, column] = np.interp(times, times[valid], values[valid])
        else:
            coords[:, column] = 0.0

    duration = float(times[-1] - times[0])
    grid = times[0] + np.arange(int(np.floor(duration * TARGET_FPS)) + 1) / TARGET_FPS
    output = np.empty((len(grid), coords.shape[1]), dtype="float32")
    for column in range(coords.shape[1]):
        output[:, column] = np.interp(grid, times, coords[:, column])
    return output


def prepare(args: argparse.Namespace) -> None:
    frame_paths = [resolve_file(args, filename) for filename in DATA_FILES]
    membership = load_fold_membership(resolve_file(args, FOLDS_FILE), args.fold)
    table = load_frames(frame_paths)

    feature_columns = [
        name for name in table.schema.names if name not in METADATA_COLUMNS
    ]
    if len(feature_columns) != 66:
        raise ValueError(f"expected 66 pose columns, found {len(feature_columns)}")

    rows_by_sequence: dict[str, list[int]] = defaultdict(list)
    sequence_ids = table["sequence_id"].to_pylist()
    for index, sequence_id in enumerate(sequence_ids):
        rows_by_sequence[str(sequence_id)].append(index)

    participant_ids = table["participant_id"].to_pylist()
    labels = table["maneuver_label"].to_pylist()
    eligible = table["stage2_eligible"].to_pylist()
    times = table["relative_time_s"].to_numpy(zero_copy_only=False).astype("float64")
    coordinates = np.column_stack(
        [
            table[column].to_numpy(zero_copy_only=False).astype("float32")
            for column in feature_columns
        ]
    )
    samples: dict[str, list[tuple[np.ndarray, str]]] = {
        subset: [] for subset in ("train", "val", "test")
    }

    for indices in rows_by_sequence.values():
        first = indices[0]
        participant_id = str(participant_ids[first])
        subset = membership.get(participant_id)
        if subset is None:
            raise ValueError(f"participant missing from fold {args.fold}: {participant_id}")
        label = str(labels[first])
        if label not in LABELS or not bool(eligible[first]):
            continue
        sequence = fill_and_resample(times[indices], coordinates[indices].copy())
        if len(sequence) < WINDOW_FRAMES:
            continue
        step = STEP_FRAMES[label]
        for start in range(0, len(sequence) - WINDOW_FRAMES + 1, step):
            samples[subset].append(
                (sequence[start : start + WINDOW_FRAMES].copy(), label)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subset, subset_samples in samples.items():
        output = args.output_dir / f"fold_{args.fold}_{subset}_data.pkl"
        with output.open("wb") as handle:
            pickle.dump(subset_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        counts = Counter(label for _window, label in subset_samples)
        print(f"{output}: {len(subset_samples)} windows {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    prepare(parse_args())
