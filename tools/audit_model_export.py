#!/usr/bin/env python3
"""Export packaged TorchScript models to ONNX and verify numerical equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model_package"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    input_shape: tuple[int, int]
    output_names: tuple[str, ...]
    classes: tuple[str, ...]


SPECS = (
    ModelSpec(
        "head",
        "head_model.pt",
        (12, 66),
        ("logits", "feature"),
        ("Left_Look", "Right_Look", "neutral_head"),
    ),
    ModelSpec(
        "upper",
        "upper_model.pt",
        (12, 66),
        ("logits", "feature"),
        (
            "Upper_Limb_Left_Rotation",
            "Upper_Limb_Right_Rotation",
            "neutral_upper_limb",
        ),
    ),
    ModelSpec(
        "leg",
        "leg_model.pt",
        (20, 66),
        ("logits", "feature"),
        ("Pedaling", "neutral_leg"),
    ),
    ModelSpec(
        "stage2",
        "stage2_maneuver_model.pt",
        (120, 640),
        ("logits",),
        ("straight", "yield", "overtake"),
    ),
)
FRAME_METADATA_COLUMNS = {
    "sequence_id",
    "participant_id",
    "frame_index",
    "relative_time_s",
    "fixed_fps",
    "route_anchor",
    "maneuver_label",
    "stage2_eligible",
}
EQUIVALENCE_ATOL = 1e-4
EQUIVALENCE_RTOL = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def outputs(value) -> tuple[torch.Tensor, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def export_model(spec: ModelSpec, output: Path, opset: int) -> dict:
    source = MODEL_DIR / spec.filename
    model = torch.jit.load(str(source), map_location="cpu").eval()
    example = torch.zeros((1, *spec.input_shape), dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch"}}
    dynamic_axes.update({name: {0: "batch"} for name in spec.output_names})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            model,
            example,
            str(output),
            input_names=["input"],
            output_names=list(spec.output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    return {
        "source": spec.filename,
        "source_sha256": sha256(source),
        "source_bytes": source.stat().st_size,
        "onnx": output.name,
        "onnx_sha256": sha256(output),
        "onnx_bytes": output.stat().st_size,
        "forward_schema": str(model.forward.schema),
        "torch_operators": sorted({node.kind() for node in model.inlined_graph.nodes()}),
        "onnx_operators": operators,
        "export_warnings": sorted({str(item.message) for item in caught}),
    }


def compare_case(
    spec: ModelSpec,
    model,
    session: ort.InferenceSession,
    values: np.ndarray,
    case_name: str,
) -> dict:
    with torch.no_grad():
        reference = outputs(model(torch.from_numpy(values)))
    candidate = session.run(None, {"input": values})
    output_metrics = []
    passed = True
    for name, expected, actual in zip(spec.output_names, reference, candidate):
        expected_array = expected.detach().cpu().numpy()
        difference = np.abs(expected_array - actual)
        close = bool(
            np.allclose(
                expected_array,
                actual,
                rtol=EQUIVALENCE_RTOL,
                atol=EQUIVALENCE_ATOL,
            )
        )
        passed = passed and close
        output_metrics.append(
            {
                "name": name,
                "shape": list(actual.shape),
                "max_abs_error": float(difference.max(initial=0.0)),
                "mean_abs_error": float(difference.mean()),
                "within_rtol_1e-4_atol_1e-4": close,
            }
        )
    expected_class = np.argmax(reference[0].detach().cpu().numpy(), axis=1)
    actual_class = np.argmax(candidate[0], axis=1)
    class_equal = bool(np.array_equal(expected_class, actual_class))
    return {
        "case": case_name,
        "batch": int(values.shape[0]),
        "outputs": output_metrics,
        "class_indices_equal": class_equal,
        "passed": passed and class_equal,
    }


def audit_model(
    spec: ModelSpec,
    onnx_path: Path,
    rng: np.random.Generator,
) -> tuple[dict, object, ort.InferenceSession]:
    model = torch.jit.load(str(MODEL_DIR / spec.filename), map_location="cpu").eval()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    cases = []
    for batch in (1, 2):
        shape = (batch, *spec.input_shape)
        cases.extend(
            [
                compare_case(
                    spec,
                    model,
                    session,
                    np.zeros(shape, dtype=np.float32),
                    f"zeros_batch_{batch}",
                ),
                compare_case(
                    spec,
                    model,
                    session,
                    np.ones(shape, dtype=np.float32),
                    f"ones_batch_{batch}",
                ),
                compare_case(
                    spec,
                    model,
                    session,
                    rng.uniform(0.0, 1.0, size=shape).astype("float32"),
                    f"normalized_pose_batch_{batch}",
                ),
                compare_case(
                    spec,
                    model,
                    session,
                    rng.normal(0.0, 1.0, size=shape).astype("float32"),
                    f"normal_batch_{batch}",
                ),
            ]
        )
    return {
        "classes": list(spec.classes),
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }, model, session


def end_to_end_check(models: dict, sessions: dict, rng: np.random.Generator) -> dict:
    batch = 120
    inputs = {
        "head": rng.uniform(0.0, 1.0, size=(batch, 12, 66)).astype("float32"),
        "upper": rng.uniform(0.0, 1.0, size=(batch, 12, 66)).astype("float32"),
        "leg": rng.uniform(0.0, 1.0, size=(batch, 20, 66)).astype("float32"),
    }
    torch_features = []
    onnx_features = []
    with torch.no_grad():
        for key in ("head", "upper", "leg"):
            torch_features.append(outputs(models[key](torch.from_numpy(inputs[key])))[1])
            onnx_features.append(
                torch.from_numpy(sessions[key].run(None, {"input": inputs[key]})[1])
            )
        torch_stage2_input = torch.cat(torch_features, dim=1).unsqueeze(0)
        onnx_stage2_input = torch.cat(onnx_features, dim=1).unsqueeze(0).numpy()
        torch_logits = models["stage2"](torch_stage2_input).detach().cpu().numpy()
    onnx_logits = sessions["stage2"].run(None, {"input": onnx_stage2_input})[0]
    difference = np.abs(torch_logits - onnx_logits)
    class_equal = bool(
        np.array_equal(
            np.argmax(torch_logits, axis=1),
            np.argmax(onnx_logits, axis=1),
        )
    )
    return {
        "stage1_timepoints": batch,
        "stage2_input_shape": list(onnx_stage2_input.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "class_indices_equal": class_equal,
        "passed": bool(
            np.allclose(
                torch_logits,
                onnx_logits,
                rtol=EQUIVALENCE_RTOL,
                atol=EQUIVALENCE_ATOL,
            )
            and class_equal
        ),
    }


def stage2_from_windows(models: dict, sessions: dict, windows: dict) -> dict:
    torch_features = []
    onnx_features = []
    with torch.no_grad():
        for key in ("head", "upper", "leg"):
            torch_features.append(outputs(models[key](torch.from_numpy(windows[key])))[1])
            onnx_features.append(
                torch.from_numpy(sessions[key].run(None, {"input": windows[key]})[1])
            )
        torch_stage2_input = torch.cat(torch_features, dim=1).unsqueeze(0)
        onnx_stage2_input = torch.cat(onnx_features, dim=1).unsqueeze(0).numpy()
        torch_logits = models["stage2"](torch_stage2_input).detach().cpu().numpy()
    onnx_logits = sessions["stage2"].run(None, {"input": onnx_stage2_input})[0]
    difference = np.abs(torch_logits - onnx_logits)
    class_equal = bool(
        np.array_equal(
            np.argmax(torch_logits, axis=1),
            np.argmax(onnx_logits, axis=1),
        )
    )
    return {
        "stage2_input_shape": list(onnx_stage2_input.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "class_indices_equal": class_equal,
        "passed": bool(
            np.allclose(
                torch_logits,
                onnx_logits,
                rtol=EQUIVALENCE_RTOL,
                atol=EQUIVALENCE_ATOL,
            )
            and class_equal
        ),
    }


def real_dataset_check(
    dataset_dir: Path,
    models: dict,
    sessions: dict,
) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    paths = sorted((dataset_dir / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no split Parquet files found under {dataset_dir}")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    feature_columns = [
        name for name in table.schema.names if name not in FRAME_METADATA_COLUMNS
    ]
    if len(feature_columns) != 66:
        raise ValueError(f"expected 66 pose columns, found {len(feature_columns)}")
    order = pc.sort_indices(
        table,
        sort_keys=[("sequence_id", "ascending"), ("frame_index", "ascending")],
    )
    table = pc.take(table, order)
    sequence_ids = table["sequence_id"].to_pylist()
    coordinates = np.column_stack(
        [
            table[column].to_numpy(zero_copy_only=False).astype("float32")
            for column in feature_columns
        ]
    )
    grouped: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(sequence_ids):
        grouped.setdefault(str(sequence_id), []).append(index)
    missing_values = int((~np.isfinite(coordinates)).sum())
    sequences = []
    for sequence_id, indices in sorted(grouped.items()):
        if len(indices) < 20:
            continue
        sequence = coordinates[indices].copy()
        positions = np.arange(len(sequence), dtype=np.float64)
        for column in range(sequence.shape[1]):
            values = sequence[:, column]
            valid = np.isfinite(values)
            if valid.any():
                sequence[:, column] = np.interp(
                    positions,
                    positions[valid],
                    values[valid],
                )
            else:
                sequence[:, column] = 0.0
        sequences.append((sequence_id, sequence))
    selected = sequences[:32]
    if not selected:
        raise ValueError("public dataset contains no sequence with at least 20 frames")

    checks = {}
    by_key = {spec.key: spec for spec in SPECS}
    for key in ("head", "upper", "leg"):
        length = by_key[key].input_shape[0]
        values = np.stack([sequence[:length] for _sequence_id, sequence in selected])
        checks[key] = compare_case(
            by_key[key],
            models[key],
            sessions[key],
            values.astype("float32"),
            "public_hf_pose_windows",
        )

    long_sequence_id, long_sequence = next(
        (
            (sequence_id, sequence)
            for sequence_id, sequence in sequences
            if len(sequence) >= 139
        ),
        (None, None),
    )
    if long_sequence is None:
        raise ValueError("public dataset contains no sequence with at least 139 frames")
    windows = {
        "head": np.stack([long_sequence[start : start + 12] for start in range(120)]),
        "upper": np.stack([long_sequence[start : start + 12] for start in range(120)]),
        "leg": np.stack([long_sequence[start : start + 20] for start in range(120)]),
    }
    pipeline = stage2_from_windows(models, sessions, windows)
    pipeline["sequence_id"] = long_sequence_id
    return {
        "dataset_dir": str(dataset_dir),
        "frame_rows": int(table.num_rows),
        "sequences": len(grouped),
        "sampled_sequences": len(selected),
        "missing_coordinate_values_before_imputation": missing_values,
        "imputation": "per-sequence linear interpolation with edge fill; all-missing columns become zero",
        "stage1": checks,
        "stage2_pipeline": pipeline,
        "passed": bool(
            all(check["passed"] for check in checks.values())
            and pipeline["passed"]
        ),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# TorchScript to ONNX Export Equivalence Audit",
        "",
        f"- Overall result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Seed: `{report['seed']}`",
        f"- ONNX opset: `{report['opset']}`",
        f"- Equivalence tolerance: `atol={report['equivalence_tolerance']['absolute']}, "
        f"rtol={report['equivalence_tolerance']['relative']}`",
        f"- PyTorch: `{report['environment']['torch']}`",
        f"- ONNX: `{report['environment']['onnx']}`",
        f"- ONNX Runtime: `{report['environment']['onnxruntime']}`",
        "",
        "## Model results",
        "",
        "| Model | Source SHA-256 | ONNX SHA-256 | Max absolute error | Class agreement | Result |",
        "|---|---|---|---:|---|---|",
    ]
    for key, result in report["models"].items():
        max_error = max(
            metric["max_abs_error"]
            for case in result["equivalence"]["cases"]
            for metric in case["outputs"]
        )
        agreement = all(
            case["class_indices_equal"] for case in result["equivalence"]["cases"]
        )
        lines.append(
            f"| {key} | `{result['export']['source_sha256']}` | "
            f"`{result['export']['onnx_sha256']}` | {max_error:.8g} | "
            f"{agreement} | {'PASS' if result['equivalence']['passed'] else 'FAIL'} |"
        )
    end_to_end = report["end_to_end"]
    lines.extend(
        [
            "",
            "## End-to-end Stage1 feature to Stage2 check",
            "",
            f"- Input shape: `{end_to_end['stage2_input_shape']}`",
            f"- Maximum absolute error: `{end_to_end['max_abs_error']:.8g}`",
            f"- Class indices equal: `{end_to_end['class_indices_equal']}`",
            f"- Result: **{'PASS' if end_to_end['passed'] else 'FAIL'}**",
            "",
            "## Boundary",
            "",
            "This audit establishes CPU numerical equivalence for the tested inputs.",
            "It does not establish TensorRT compatibility, FP16/INT8 equivalence,",
            "target-device latency, or dataset-level accuracy equivalence.",
            "",
        ]
    )
    if "real_dataset" in report:
        real = report["real_dataset"]
        real_stage1_errors = {
            key: max(
                output["max_abs_error"]
                for output in check["outputs"]
            )
            for key, check in real["stage1"].items()
        }
        lines.extend(
            [
                "## Public Hugging Face pose-data check",
                "",
                f"- Frame rows: `{real['frame_rows']}`",
                f"- Sequences: `{real['sequences']}`",
                f"- Sampled Stage1 sequences: `{real['sampled_sequences']}`",
                f"- Missing coordinate values before imputation: "
                f"`{real['missing_coordinate_values_before_imputation']}`",
                f"- Imputation: {real['imputation']}",
                f"- Stage1 maximum absolute errors: "
                f"`head={real_stage1_errors['head']:.8g}, "
                f"upper={real_stage1_errors['upper']:.8g}, "
                f"leg={real_stage1_errors['leg']:.8g}`",
                f"- Stage2 source sequence: `{real['stage2_pipeline']['sequence_id']}`",
                f"- Stage2 maximum absolute error: "
                f"`{real['stage2_pipeline']['max_abs_error']:.8g}`",
                f"- Result: **{'PASS' if real['passed'] else 'FAIL'}**",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Optional local snapshot of the public Hugging Face dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    release_root = (ROOT / "release").resolve()
    if release_root not in output_dir.parents:
        raise RuntimeError("--output-dir must be under release/")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    report = {
        "seed": args.seed,
        "opset": args.opset,
        "equivalence_tolerance": {
            "absolute": EQUIVALENCE_ATOL,
            "relative": EQUIVALENCE_RTOL,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
        },
        "models": {},
    }
    models = {}
    sessions = {}
    source_hashes_before = {
        spec.key: sha256(MODEL_DIR / spec.filename) for spec in SPECS
    }
    for spec in SPECS:
        onnx_path = output_dir / f"{spec.key}.onnx"
        export = export_model(spec, onnx_path, args.opset)
        equivalence, model, session = audit_model(spec, onnx_path, rng)
        report["models"][spec.key] = {
            "export": export,
            "equivalence": equivalence,
        }
        models[spec.key] = model
        sessions[spec.key] = session

    report["end_to_end"] = end_to_end_check(models, sessions, rng)
    if args.dataset_dir is not None:
        report["real_dataset"] = real_dataset_check(
            args.dataset_dir.resolve(),
            models,
            sessions,
        )
    source_hashes_after = {
        spec.key: sha256(MODEL_DIR / spec.filename) for spec in SPECS
    }
    report["source_models_unchanged"] = source_hashes_before == source_hashes_after
    report["passed"] = bool(
        report["source_models_unchanged"]
        and all(
            item["equivalence"]["passed"] for item in report["models"].values()
        )
        and report["end_to_end"]["passed"]
        and report.get("real_dataset", {"passed": True})["passed"]
    )
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "AUDIT_REPORT.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": report["passed"],
        "output_dir": str(output_dir),
        "source_models_unchanged": report["source_models_unchanged"],
        "end_to_end": report["end_to_end"],
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
