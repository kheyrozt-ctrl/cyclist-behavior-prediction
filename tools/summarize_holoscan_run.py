#!/usr/bin/env python3
"""Summarize structured timing output from a Holoscan JSONL run."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values):
    return {
        "samples": len(values),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.jsonl.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise SystemExit("No records found")
    if any("timing_ms" not in record for record in records):
        raise SystemExit("Input records do not contain timing_ms")

    first_timestamp = float(records[0]["timestamp"])
    last_timestamp = float(records[-1]["timestamp"])
    span = max(0.0, last_timestamp - first_timestamp)
    pose_records = [record for record in records if record["pose_ok"]]
    first_stage1 = next(
        (record["frame_index"] for record in records if record["progress"][0] > 0),
        None,
    )
    first_stage2 = next(
        (
            record["frame_index"]
            for record in records
            if "collecting" not in record["lines"][3]
        ),
        None,
    )
    timing_keys = ("pose", "predictor", "operator", "graph_since_source")
    report = {
        "path": args.jsonl.as_posix(),
        "records": len(records),
        "frame_range": [
            records[0]["frame_index"],
            records[-1]["frame_index"],
        ],
        "pose_ok": len(pose_records),
        "pose_ok_fraction": round(len(pose_records) / len(records), 6),
        "timestamp_span_s": round(span, 6),
        "processed_fps": (
            round((len(records) - 1) / span, 6) if span > 0 else None
        ),
        "first_stage1_frame": first_stage1,
        "first_stage2_frame": first_stage2,
        "final_progress": records[-1]["progress"],
        "timing_ms": {
            key: distribution(
                [float(record["timing_ms"][key]) for record in records]
            )
            for key in timing_keys
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
