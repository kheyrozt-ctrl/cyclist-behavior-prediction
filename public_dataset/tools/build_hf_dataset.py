#!/usr/bin/env python3
"""Build and validate the public cyclist 2D-keypoint Hugging Face release."""

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import random
import re
import secrets
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "open_source" / "training" / "intersection_intention_legacy"
SOURCE_DIR = SOURCE_ROOT / "Full33SkeletonData"
NOTES_PATH = SOURCE_ROOT / "GestureDataset" / "Simulator Trial Notes.csv"
DEFAULT_OUTPUT = DATASET_ROOT / "release" / "hf" / "cyclist-intention-2d-keypoints"
SECRET_PATH = DATASET_ROOT / ".release-secrets" / "vru_hmac.key"
FILENAME_RE = re.compile(
    r"^video(?P<scenario>\d+[a-z])_(?P<token>\d+)_(?P<source>\d+)_.*_skeleton\.csv$"
)
FORBIDDEN_NAMES = {"timestamp", "source_file", "original_filename", "raw_participant_id"}
SPLIT_SEED = 20260623
MIN_SEQUENCE_FRAMES = 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--license-approved-by", required=True)
    parser.add_argument("--repo-id", default="Kheyro/cyclist-intention-2d-keypoints")
    return parser.parse_args()


def load_or_create_key():
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SECRET_PATH.exists():
        key = SECRET_PATH.read_bytes()
    else:
        key = secrets.token_bytes(32)
        with SECRET_PATH.open("xb") as handle:
            handle.write(key)
    if len(key) < 32:
        raise RuntimeError("Pseudonymization key must contain at least 32 bytes")
    return key


def public_id(key, namespace, value, size=12):
    digest = hmac.new(key, f"{namespace}:{value}".encode(), hashlib.sha256).hexdigest()
    return digest[:size]


def load_trial_metadata():
    rows = []
    with NOTES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    source_to_participant = defaultdict(set)
    exact = {}
    for row in rows:
        source_to_participant[row["source_file"]].add(int(row["participant_id"]))
        key = (row["source_file"], row["scenario"])
        if key in exact:
            raise RuntimeError(f"Duplicate trial metadata: {key}")
        exact[key] = row
    ambiguous = {key: values for key, values in source_to_participant.items() if len(values) != 1}
    if ambiguous:
        raise RuntimeError(f"Ambiguous source-to-participant mappings: {ambiguous}")
    return {key: next(iter(values)) for key, values in source_to_participant.items()}, exact


def discover_sequences(source_to_participant, exact_notes, key):
    sequences = []
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            raise RuntimeError(f"Unexpected source filename: {path.name}")
        source = match["source"]
        scenario = match["scenario"]
        if source not in source_to_participant:
            raise RuntimeError(f"Source session has no participant metadata: {source}")
        note = exact_notes.get((source, scenario))
        if note is None:
            raise RuntimeError(f"Trial has no exact metadata row: {path.name}")
        if note["use"].strip().upper() != "Y":
            continue
        raw_participant = source_to_participant[source]
        sequences.append({
            "path": path,
            "scenario": scenario,
            "raw_participant": raw_participant,
            "participant_id": "vru_" + public_id(key, "participant", raw_participant),
            "sequence_id": "seq_" + public_id(key, "sequence", f"{source}:{scenario}", 16),
        })
    if not sequences:
        raise RuntimeError("No approved skeleton sequences were found")
    return sequences


def assign_splits(sequences):
    participants = sorted({item["raw_participant"] for item in sequences})
    shuffled = participants[:]
    random.Random(SPLIT_SEED).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.15))
    validation_count = max(1, round(len(shuffled) * 0.15))
    test = set(shuffled[:test_count])
    validation = set(shuffled[test_count:test_count + validation_count])
    train = set(shuffled[test_count + validation_count:])
    assignment = {participant: "train" for participant in train}
    assignment.update({participant: "validation" for participant in validation})
    assignment.update({participant: "test" for participant in test})
    for item in sequences:
        item["split"] = assignment[item["raw_participant"]]
    return assignment


def normalized_pose_columns(source_header):
    if len(source_header) != 67 or source_header[0].strip().lower() != "timestamp":
        raise RuntimeError(f"Expected timestamp + 66 pose columns, got {len(source_header)} columns")
    result = []
    for name in source_header[1:]:
        clean = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        if not clean:
            raise RuntimeError(f"Invalid pose column: {name!r}")
        result.append(clean)
    if len(set(result)) != 66:
        raise RuntimeError("Pose column names are not unique after normalization")
    return result


def arrow_schema(pose_columns):
    fields = [
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("participant_id", pa.string(), nullable=False),
        pa.field("scenario_code", pa.string(), nullable=False),
        pa.field("frame_index", pa.int32(), nullable=False),
    ]
    fields.extend(pa.field(name, pa.float32()) for name in pose_columns)
    return pa.schema(fields)


def write_parquet(sequences, output):
    data_dir = output / "data"
    data_dir.mkdir(parents=True)
    first_path = sequences[0]["path"]
    with first_path.open("r", encoding="utf-8-sig", newline="") as handle:
        pose_columns = normalized_pose_columns(next(csv.reader(handle)))
    schema = arrow_schema(pose_columns)
    writers = {
        split: pq.ParquetWriter(
            data_dir / f"{split}-00000-of-00001.parquet",
            schema,
            compression="zstd",
            use_dictionary=["sequence_id", "participant_id", "scenario_code"],
        )
        for split in ("train", "validation", "test")
    }
    sequence_rows = []
    excluded_rows = []
    split_frames = Counter()
    try:
        for index, item in enumerate(sequences, 1):
            columns = {field.name: [] for field in schema}
            missing_values = 0
            with item["path"].open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                current_pose_columns = normalized_pose_columns(next(reader))
                if current_pose_columns != pose_columns:
                    raise RuntimeError(f"Schema mismatch: {item['path'].name}")
                for frame_index, row in enumerate(reader):
                    if len(row) != 67:
                        raise RuntimeError(
                            f"Malformed row in {item['path'].name}: expected 67, got {len(row)}"
                        )
                    columns["sequence_id"].append(item["sequence_id"])
                    columns["participant_id"].append(item["participant_id"])
                    columns["scenario_code"].append(item["scenario"])
                    columns["frame_index"].append(frame_index)
                    for name, value in zip(pose_columns, row[1:]):
                        value = value.strip()
                        if value == "":
                            columns[name].append(float("nan"))
                            missing_values += 1
                        else:
                            columns[name].append(float(value))
            frame_count = len(columns["frame_index"])
            if frame_count == 0:
                raise RuntimeError(f"Empty sequence: {item['path'].name}")
            if frame_count < MIN_SEQUENCE_FRAMES:
                excluded_rows.append({
                    "sequence_id": item["sequence_id"],
                    "reason": "shorter_than_minimum_sequence_length",
                    "frame_count": frame_count,
                    "minimum_frames": MIN_SEQUENCE_FRAMES,
                })
                print(
                    f"[{index:03d}/{len(sequences)}] EXCLUDED: {item['sequence_id']} "
                    f"({frame_count} < {MIN_SEQUENCE_FRAMES} frames)"
                )
                continue
            table = pa.Table.from_pydict(columns, schema=schema)
            writers[item["split"]].write_table(table, row_group_size=4096)
            split_frames[item["split"]] += frame_count
            sequence_rows.append({
                "sequence_id": item["sequence_id"],
                "participant_id": item["participant_id"],
                "split": item["split"],
                "scenario_code": item["scenario"],
                "frame_count": frame_count,
                "missing_coordinate_values": missing_values,
            })
            print(f"[{index:03d}/{len(sequences)}] {item['split']}: {item['sequence_id']} ({frame_count} frames)")
    finally:
        for writer in writers.values():
            writer.close()
    with (output / "sequences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    return schema, sequence_rows, split_frames, excluded_rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_docs(
    output, repo_id, approved_by, key, schema, sequence_rows, split_frames, excluded_rows
):
    split_participants = defaultdict(set)
    split_sequences = Counter()
    scenarios = Counter()
    missing = 0
    for row in sequence_rows:
        split_participants[row["split"]].add(row["participant_id"])
        split_sequences[row["split"]] += 1
        scenarios[row["scenario_code"]] += 1
        missing += row["missing_coordinate_values"]
    card = f"""---
license: cc-by-4.0
task_categories:
- time-series-forecasting
- other
language:
- en
tags:
- cyclist
- vulnerable-road-users
- pose-estimation
- intention-prediction
- traffic-safety
pretty_name: Cyclist Intention 2D Keypoints
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
---

# Cyclist Intention 2D Keypoints

This dataset contains de-identified 2D cyclist pose sequences captured in a
VR-based bicycle simulator for cyclist maneuver/intention research at
unsignalized intersections. It contains only skeleton coordinates and release
metadata. Raw video, survey responses, original timestamps, source filenames,
session identifiers, and direct participant identifiers are excluded.

## Dataset summary

- 29 participants represented in the released skeleton subset
- source study reports 31 participants; local trial metadata lists 30
- {len(sequence_rows)} released pose sequences
- 33 MediaPipe/GHUM landmarks with normalized x/y coordinates (66 values)
- nominal source capture rate: 30 FPS
- participant-disjoint train/validation/test splits
- license: Creative Commons Attribution 4.0 International (CC BY 4.0)

This is the existing 2D intersection dataset. It is not the proposal's future
60–120 Hz 3D CAVE dataset with synchronized scene context.

## Splits

| Split | Participants | Sequences | Frames |
|---|---:|---:|---:|
| Train | {len(split_participants['train'])} | {split_sequences['train']} | {split_frames['train']} |
| Validation | {len(split_participants['validation'])} | {split_sequences['validation']} | {split_frames['validation']} |
| Test | {len(split_participants['test'])} | {split_sequences['test']} | {split_frames['test']} |

Participants never cross split boundaries. The split seed and exact membership
are recorded in `split_manifest.json`.

## Fields

- `sequence_id`: release-scoped unlinkable sequence pseudonym
- `participant_id`: release-scoped unlinkable participant grouping key
- `scenario_code`: simulator scenario code; retained as an opaque experimental condition
- `frame_index`: zero-based order within a sequence; original time values are removed
- 66 pose columns: lowercase landmark name plus `_x` or `_y`

The original public grouping IDs are HMAC-derived with a secret key that is not
included in the release. They support leakage-safe grouping but are not intended
for linkage to source identities.

## Intended uses

- cyclist pose and temporal-behavior modeling
- participant-independent validation
- robustness, missing-data, and sequence-model research
- reproducibility studies related to cyclist intention prediction

## Out-of-scope uses

- biometric identification or re-identification
- surveillance, policing, insurance, employment, or eligibility decisions
- safety-critical vehicle control without independent validation
- claims about real-world demographic representativeness

## Limitations and privacy

The data were collected in a simulator and may not generalize to natural traffic.
Pose trajectories can retain behavioral signatures; removal of direct identifiers
does not eliminate every theoretical re-identification risk. Users must not try
to identify participants or combine the dataset with external identity sources.
Scenario codes are not maneuver labels unless supported by separate study metadata.

## Source and citation

Tian Zheng, Yunfei Zhang, Mathias Pechinger, Johannes Lindner, and Klaus
Bogenberger. *Cyclist Maneuver Prediction at Unsignalized Intersection using a
VR-based Bike Simulator*. EasyChair Preprint, 2025.

## License

Licensed under CC BY 4.0. See `LICENSE.md`. Attribution must identify the dataset
title, authors/source study, license, and changes made.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    license_text = """# Creative Commons Attribution 4.0 International

This dataset is licensed under the Creative Commons Attribution 4.0
International license (CC BY 4.0).

License deed: https://creativecommons.org/licenses/by/4.0/
Legal code: https://creativecommons.org/licenses/by/4.0/legalcode

You are free to share and adapt the material for any purpose, including
commercially, provided that appropriate credit is given, a link to the license
is supplied, and changes are indicated. No additional restrictions may be
applied. See the legal code for the complete terms.
"""
    (output / "LICENSE.md").write_text(license_text, encoding="utf-8")
    participant_sets = {split: sorted(values) for split, values in split_participants.items()}
    split_manifest = {
        "version": "1.0.0",
        "split_method": "participant-disjoint deterministic shuffle",
        "split_seed": SPLIT_SEED,
        "participants": participant_sets,
        "sequence_counts": dict(split_sequences),
        "frame_counts": dict(split_frames),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    privacy_report = {
        "release_scope": "2D skeleton coordinates and minimized metadata only",
        "removed": [
            "raw video", "survey data", "original timestamps", "source filenames",
            "source session IDs", "direct participant IDs", "free-text notes",
        ],
        "public_grouping": "release-scoped HMAC participant and sequence IDs",
        "key_in_release": False,
        "key_version_sha256_prefix": hashlib.sha256(key).hexdigest()[:12],
        "participant_split_overlap": False,
        "missing_coordinate_values": missing,
        "residual_risk": (
            "Pose trajectories may retain behavioral signatures. The release prohibits "
            "re-identification and documents this limitation."
        ),
        "license_approval_record": approved_by,
    }
    (output / "privacy_report.json").write_text(
        json.dumps(privacy_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality_report = {
        "minimum_sequence_frames": MIN_SEQUENCE_FRAMES,
        "released_sequence_count": len(sequence_rows),
        "excluded_sequence_count": len(excluded_rows),
        "exclusions": excluded_rows,
        "missing_coordinate_values": missing,
        "schema_validated": True,
    }
    (output / "quality_report.json").write_text(
        json.dumps(quality_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    citation = """cff-version: 1.2.0
message: "If you use this dataset, please cite the source study."
title: "Cyclist Intention 2D Keypoints"
type: dataset
license: CC-BY-4.0
version: 1.0.0
authors:
  - family-names: Zheng
    given-names: Tian
  - family-names: Zhang
    given-names: Yunfei
  - family-names: Pechinger
    given-names: Mathias
  - family-names: Lindner
    given-names: Johannes
  - family-names: Bogenberger
    given-names: Klaus
date-released: 2026-06-23
repository-code: "https://huggingface.co/datasets/""" + repo_id + "\n"
    (output / "CITATION.cff").write_text(citation, encoding="utf-8")
    manifest = {
        "dataset": repo_id,
        "version": "1.0.0",
        "license": "CC-BY-4.0",
        "license_approved_by": approved_by,
        "discovered_source_sequence_count": len(sequence_rows) + len(excluded_rows),
        "released_sequence_count": len(sequence_rows),
        "excluded_sequence_count": len(excluded_rows),
        "public_participant_count": len({row["participant_id"] for row in sequence_rows}),
        "schema": [{"name": field.name, "type": str(field.type)} for field in schema],
        "scenario_sequence_counts": dict(sorted(scenarios.items())),
        "files": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "dataset_manifest.json":
            manifest["files"][path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_release(output):
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = output / relative
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Checksum validation failed: {relative}")
    split_participants = {}
    schemas = []
    for split in ("train", "validation", "test"):
        path = output / "data" / f"{split}-00000-of-00001.parquet"
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        forbidden = names & FORBIDDEN_NAMES
        if forbidden:
            raise RuntimeError(f"Forbidden columns in {split}: {sorted(forbidden)}")
        schemas.append(parquet.schema_arrow)
        table = pq.read_table(path, columns=["participant_id", "sequence_id", "frame_index"])
        split_participants[split] = set(table.column("participant_id").to_pylist())
        if table.num_rows == 0:
            raise RuntimeError(f"Empty split: {split}")
    if not all(schema.equals(schemas[0]) for schema in schemas[1:]):
        raise RuntimeError("Split schemas do not match")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_participants[left] & split_participants[right]
        if overlap:
            raise RuntimeError(f"Participant leakage between {left} and {right}: {overlap}")
    release_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.glob("*") if path.is_file()
    )
    if re.search(r"video\d|1568\d{6}|202408\d{2}", release_text, re.IGNORECASE):
        raise RuntimeError("Potential source identifier leaked into release metadata")
    print("Validation passed: checksums, schemas, privacy columns, and participant isolation")


def main():
    args = parse_args()
    output = args.output.resolve()
    release_root = (ROOT / "release").resolve()
    if release_root not in output.parents:
        raise RuntimeError(f"Output must remain under {release_root}")
    if output.exists():
        raise RuntimeError(f"Output already exists; remove it explicitly before rebuilding: {output}")
    output.mkdir(parents=True)
    key = load_or_create_key()
    source_to_participant, exact_notes = load_trial_metadata()
    sequences = discover_sequences(source_to_participant, exact_notes, key)
    assignment = assign_splits(sequences)
    if len(assignment) < 2:
        raise RuntimeError(f"At least two participants are required, got {len(assignment)}")
    schema, sequence_rows, split_frames, excluded_rows = write_parquet(sequences, output)
    write_release_docs(
        output, args.repo_id, args.license_approved_by, key,
        schema, sequence_rows, split_frames, excluded_rows,
    )
    validate_release(output)
    print(f"Release package created: {output}")


if __name__ == "__main__":
    main()
