#!/usr/bin/env python3
"""Build the anonymized V5 cyclist-behavior Hugging Face dataset release."""

import argparse, csv, hashlib, hmac, json, re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "0cyclistprediction_v5"
SKELETONS = SRC / "Full33SkeletonData" / "stage1_scene_clips_20260622"
VC = SRC / "Gesture2Manuever_Prediction" / "VideoCrop"
MD = SRC / "Gesture2Manuever_Prediction" / "ManueverPrediction_Combined" / "ManueverDataset"
OUT = ROOT / "release" / "hf" / "cyclist-intention-2d-keypoints-v2"
KEY = ROOT / ".release-secrets" / "vru_hmac.key"
SK_RE = re.compile(r"^Participant_(\d+)_\d+_.+_cam2rgbfixed_skeleton\.csv$")


def pid(key, value):
    return "vru_" + hmac.new(key, f"v5-participant:{value}".encode(), hashlib.sha256).hexdigest()[:12]


def sid(key, value):
    return "seq_" + hmac.new(key, f"v5-sequence:{value}".encode(), hashlib.sha256).hexdigest()[:16]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def fold_participants():
    folds = {}
    for fold in range(1, 6):
        subsets = {}
        for subset in ("train", "val", "test"):
            lines = (VC / f"fold_{fold}_{subset}_files.txt").read_text(encoding="utf-8-sig").splitlines()
            subsets[subset] = {int(re.match(r"Participant_(\d+)_", x).group(1)) for x in lines if x.strip()}
        if any(subsets[a] & subsets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError(f"Participant leakage in fold {fold}")
        folds[fold] = subsets
    return folds


def metadata():
    with (VC / "gesture_crop_info.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows: grouped[row["base_clip_id"]].append(row)
    eligible = set()
    for subset in ("train", "val", "test"):
        with (MD / f"fold_1_{subset}_data.csv").open(encoding="utf-8-sig", newline="") as f:
            eligible.update(row["skeleton_id"] for row in csv.DictReader(f))
    return grouped, eligible


def pose_names(header):
    if len(header) != 67 or header[0].strip().lower() != "timestamp":
        raise RuntimeError(f"Expected timestamp + 66 coordinates, got {len(header)}")
    names = [re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_") for x in header[1:]]
    if len(set(names)) != 66: raise RuntimeError("Non-unique normalized pose columns")
    return names


def build(output, repo_id, approved_by):
    key = KEY.read_bytes()
    folds = fold_participants()
    grouped, eligible = metadata()
    files = sorted(SKELETONS.glob("*.csv"))
    if len(files) != 559: raise RuntimeError(f"Expected 559 skeleton clips, got {len(files)}")
    with files[0].open(encoding="utf-8-sig", newline="") as f: poses = pose_names(next(csv.reader(f)))
    schema = pa.schema([
        pa.field("sequence_id", pa.string(), False), pa.field("participant_id", pa.string(), False),
        pa.field("frame_index", pa.int32(), False), pa.field("relative_time_s", pa.float32(), False),
        pa.field("fixed_fps", pa.float32(), False), pa.field("route_anchor", pa.int8(), False),
        pa.field("maneuver_label", pa.string(), False), pa.field("stage2_eligible", pa.bool_(), False),
        *[pa.field(x, pa.float32()) for x in poses],
    ])
    data = output / "data"; data.mkdir(parents=True)
    writers = {s: pq.ParquetWriter(data / f"{s}-00000-of-00001.parquet", schema, compression="zstd") for s in ("train", "validation", "test")}
    seq_rows, annotation_rows = [], []
    frame_counts, seq_counts = Counter(), Counter()
    public_by_raw = {raw: pid(key, raw) for raw in set().union(*folds[1].values())}
    try:
        for i, path in enumerate(files, 1):
            m = SK_RE.match(path.name)
            if not m: raise RuntimeError(f"Unexpected filename: {path.name}")
            raw_pid = int(m.group(1)); base = path.stem.removesuffix("_skeleton")
            if base not in grouped: raise RuntimeError(f"No metadata for {base}")
            rows = grouped[base]
            labels = {r["maneuver_label"].strip() for r in rows if r["maneuver_label"].strip()}
            anchors = {int(float(r["route_anchor"])) for r in rows if r["route_anchor"]}
            fpss = {round(float(r["fixed_fps"]), 6) for r in rows if r["fixed_fps"]}
            if len(labels) != 1 or len(anchors) != 1 or len(fpss) != 1:
                raise RuntimeError(f"Inconsistent clip metadata: {base}")
            raw_subset = next((s for s, ps in folds[1].items() if raw_pid in ps), None)
            if raw_subset is None: raise RuntimeError(f"Participant absent from fold 1: {raw_pid}")
            split = {"train":"train", "val":"validation", "test":"test"}[raw_subset]
            sequence_id = sid(key, base); public_pid = public_by_raw[raw_pid]
            with path.open(encoding="utf-8-sig", newline="") as f:
                rd = csv.reader(f); current = pose_names(next(rd))
                if current != poses: raise RuntimeError(f"Schema mismatch: {path.name}")
                source_rows = list(rd)
            if not source_rows: raise RuntimeError(f"Empty clip: {path.name}")
            t0 = float(source_rows[0][0]); columns = {field.name: [] for field in schema}
            for frame, row in enumerate(source_rows):
                if len(row) != 67: raise RuntimeError(f"Malformed frame: {path.name}:{frame}")
                fixed = [sequence_id, public_pid, frame, float(row[0])-t0, next(iter(fpss)), next(iter(anchors)), next(iter(labels)), base in eligible]
                for name, value in zip(list(columns)[:8], fixed): columns[name].append(value)
                for name, value in zip(poses, row[1:]): columns[name].append(float(value) if value.strip() else float("nan"))
            writers[split].write_table(pa.Table.from_pydict(columns, schema=schema), row_group_size=4096)
            n = len(source_rows); frame_counts[split] += n; seq_counts[split] += 1
            seq_rows.append({"sequence_id":sequence_id,"participant_id":public_pid,"split":split,"frame_count":n,"fixed_fps":next(iter(fpss)),"route_anchor":next(iter(anchors)),"maneuver_label":next(iter(labels)),"stage2_eligible":base in eligible})
            for r in rows:
                flag = r["use_flag"].strip().lower()
                annotation_rows.append({
                    "sequence_id": sequence_id,
                    "gesture_type": r["Gesture Type"].strip() or None,
                    "start_frame": int(float(r["Start Frame"])) if r["Start Frame"].strip() else None,
                    "end_frame": int(float(r["End Frame"])) if r["End Frame"].strip() else None,
                    "start_time_s": float(r["Start Timestamp"]) if r["Start Timestamp"].strip() else None,
                    "end_time_s": float(r["End Timestamp"]) if r["End Timestamp"].strip() else None,
                    "status": "excluded" if flag in {"exclude", "disable", "disabled"} else "active",
                    "exclusion_reason": r["Iexclude_reason"].strip() or None,
                })
            if i % 50 == 0: print(f"Processed {i}/559")
    finally:
        for w in writers.values(): w.close()
    # Metadata tables
    with (output/"sequences.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(seq_rows[0]));w.writeheader();w.writerows(seq_rows)
    pq.write_table(pa.Table.from_pylist(annotation_rows), output/"annotations.parquet", compression="zstd")
    fold_rows=[]
    for fold, subsets in folds.items():
        for subset, participants in subsets.items():
            for raw in sorted(participants): fold_rows.append({"fold":fold,"subset":"validation" if subset=="val" else subset,"participant_id":public_by_raw[raw]})
    pq.write_table(pa.Table.from_pylist(fold_rows), output/"participant_folds.parquet", compression="zstd")
    stats={"version":"2.0.0","participants":len(public_by_raw),"sequences":len(seq_rows),"frames":sum(frame_counts.values()),"sequence_counts":dict(seq_counts),"frame_counts":dict(frame_counts),"stage2_eligible_sequences":sum(r["stage2_eligible"] for r in seq_rows),"annotation_rows":len(annotation_rows)}
    card=f'''---
license: cc-by-4.0
task_categories:
- other
tags: [cyclist, vulnerable-road-users, pose-estimation, behavior-prediction, traffic-safety]
pretty_name: Cyclist Behavior 2D Keypoints V5
size_categories: [10K<n<100K]
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
# Cyclist Behavior 2D Keypoints V5

Version 2.0.0 is a complete replacement of the earlier intersection release. It contains
de-identified cyclist skeleton scene clips for bus-stop/road behavior prediction.

## Summary

- {stats['participants']} participants, {stats['sequences']} scene clips, {stats['frames']:,} frames
- 33 two-dimensional pose landmarks (66 coordinates)
- Stage1 labels: head look, upper-limb rotation, and pedaling intervals
- Stage2 labels: `straight`, `yield`, and `overtake`
- canonical Fold 1 participant-disjoint train/validation/test split
- all five participant folds in `participant_folds.parquet`

| Split | Sequences | Frames |
|---|---:|---:|
| Train | {seq_counts['train']} | {frame_counts['train']:,} |
| Validation | {seq_counts['validation']} | {frame_counts['validation']:,} |
| Test | {seq_counts['test']} | {frame_counts['test']:,} |

`annotations.parquet` contains sanitized Stage1 gesture intervals and exclusion flags.
`sequences.csv` contains clip-level maneuver labels, route anchors, FPS, and Stage2 eligibility.

## Privacy

Raw video, original filenames, calendar dates, session IDs, direct participant numbers,
and source paths are excluded. Release-scoped HMAC identifiers preserve grouping without
publishing the source mapping. Pose trajectories can retain behavioral signatures; users
must not attempt re-identification or linkage with external identity data.

## Intended use and limitations

Intended for cyclist behavior, temporal pose, and participant-independent evaluation research.
The simulator domain and variable source frame rate ({min(r['fixed_fps'] for r in seq_rows):.3f}–{max(r['fixed_fps'] for r in seq_rows):.3f} FPS)
limit direct real-world generalization. This dataset must not be used for biometric identification,
surveillance, or safety-critical control without independent validation.

## License

CC BY 4.0. Cite the dataset and indicate modifications.
'''
    (output/"README.md").write_text(card,encoding="utf-8")
    (output/"LICENSE.md").write_text("# CC BY 4.0\n\nLicensed under the Creative Commons Attribution 4.0 International license.\n\nhttps://creativecommons.org/licenses/by/4.0/legalcode\n",encoding="utf-8")
    (output/"release_stats.json").write_text(json.dumps(stats,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    privacy={"removed":["raw video","original filenames","calendar dates","session IDs","direct participant IDs","source paths"],"public_ids":"release-scoped HMAC","key_in_release":False,"license_approval":approved_by,"participant_overlap":False}
    (output/"privacy_report.json").write_text(json.dumps(privacy,indent=2)+"\n",encoding="utf-8")
    manifest={"dataset":repo_id,"version":"2.0.0","source":"0cyclistprediction_v5","stats":stats,"files":{}}
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name!="dataset_manifest.json":manifest["files"][p.relative_to(output).as_posix()]={"bytes":p.stat().st_size,"sha256":sha(p)}
    (output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    validate(output, folds)
    return stats


def validate(output, folds):
    schemas=[]
    for split in ("train","validation","test"):
        pf=pq.ParquetFile(output/"data"/f"{split}-00000-of-00001.parquet");schemas.append(pf.schema_arrow)
        forbidden={"timestamp","raw_participant_id","source_file","filename","session_id"}&set(pf.schema_arrow.names)
        if forbidden:raise RuntimeError(f"Identifier columns leaked: {forbidden}")
    if not all(x.equals(schemas[0]) for x in schemas[1:]):raise RuntimeError("Split schema mismatch")
    for fold, subsets in folds.items():
        if any(subsets[a]&subsets[b] for a,b in (("train","val"),("train","test"),("val","test"))):raise RuntimeError(f"Fold leakage: {fold}")
    text="\n".join(p.read_text(encoding="utf-8",errors="ignore") for p in output.glob("*") if p.is_file())
    if re.search(r"Participant_\d+|20260\d{3}|cam2rgbfixed",text):raise RuntimeError("Source identifier leaked into release text")
    print("V5 release validation passed")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=OUT);ap.add_argument("--repo-id",default="Kheyro/cyclist-intention-2d-keypoints");ap.add_argument("--license-approved-by",required=True);a=ap.parse_args()
    out=a.output.resolve();root=(ROOT/"release").resolve()
    if root not in out.parents:raise RuntimeError("Output must be under release/")
    if out.exists():raise RuntimeError(f"Output exists: {out}")
    out.mkdir(parents=True);stats=build(out,a.repo_id,a.license_approved_by);print(json.dumps(stats,indent=2))

if __name__=="__main__":main()
