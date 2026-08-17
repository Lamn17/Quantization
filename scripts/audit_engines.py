#!/usr/bin/env python3
"""Inspect saved TensorRT engines without rebuilding or mutating run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from quantization import _inspect_engine


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No TensorRT engines were audited")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def engine_for_run(run_dir: Path) -> Path:
    engines = sorted((run_dir / "artifacts").glob("*.engine"))
    if len(engines) != 1:
        raise ValueError(f"Expected exactly one saved engine in {run_dir}: {engines}")
    return engines[0]


def summarize_inspection(
    dataset: str,
    run_dir: Path,
    engine: Path,
    inspection_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    run_info = read_json(run_dir / "run_info.json")
    inspector_path = inspection_dir / "engine_inspector.json"
    layer_count = quantize_nodes = weight_quantize_nodes = None
    input_type = output_type = None
    if result["status"] == "completed":
        inspection = read_json(inspector_path)
        layers = inspection.get("Layers", [])
        layer_names = [
            layer if isinstance(layer, str) else json.dumps(layer, sort_keys=True)
            for layer in layers
        ]
        layer_count = len(layer_names)
        quantize_nodes = sum("QuantizeLinear" in layer for layer in layer_names)
        weight_quantize_nodes = sum(
            "weight_QuantizeLinear" in layer for layer in layer_names
        )
        io_tensors = inspection.get("I/O Tensors", [])
        inputs = [item for item in io_tensors if item.get("IOMode") == "Input"]
        outputs = [item for item in io_tensors if item.get("IOMode") == "Output"]
        input_type = ";".join(str(item.get("DataType")) for item in inputs)
        output_type = ";".join(str(item.get("DataType")) for item in outputs)
    qdq_detected = bool(quantize_nodes)
    precision = str(run_info["precision"])
    if precision == "int8" and qdq_detected:
        audit_interpretation = (
            "INT8 Q/DQ graph detected; per-layer and per-channel scheme unavailable "
            "because the engine was built with layer-name-only profiling verbosity"
        )
    elif precision == "trt_fp16" and result["status"] == "completed":
        audit_interpretation = "TensorRT FP16 control engine deserialized successfully"
    else:
        audit_interpretation = "Precision audit incomplete"
    return {
        "dataset": dataset,
        "run": run_dir.name,
        "declared_precision": precision,
        "engine": str(engine),
        "engine_sha256": sha256(engine),
        "inspection_status": result["status"],
        "inspection_method": result.get("method"),
        "layer_count": layer_count,
        "quantize_named_layers": quantize_nodes,
        "weight_quantize_named_layers": weight_quantize_nodes,
        "qdq_detected": qdq_detected,
        "input_data_type": input_type,
        "output_data_type": output_type,
        "audit_interpretation": audit_interpretation,
        "inspector_artifact": str(inspector_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    args = parser.parse_args()
    if not args.audit_id.strip() or "/" in args.audit_id or ".." in args.audit_id:
        raise ValueError("--audit-id must be a non-empty directory-safe name")
    output_root = Path(args.output_root).resolve()
    audit_root = output_root / "engine_audit" / args.audit_id
    audit_root.mkdir(parents=True, exist_ok=False)
    rows = []
    failures = []
    for dataset in args.dataset:
        dataset_root = output_root / dataset
        run_dirs = [dataset_root / "trt_fp16"] + [
            dataset_root / f"int8_random_s{seed}" for seed in range(5)
        ]
        for run_dir in run_dirs:
            try:
                engine = engine_for_run(run_dir)
                inspection_dir = audit_root / dataset / run_dir.name
                inspection_dir.mkdir(parents=True)
                result = _inspect_engine(engine, inspection_dir)
                rows.append(
                    summarize_inspection(
                        dataset, run_dir, engine, inspection_dir, result
                    )
                )
                if result["status"] != "completed":
                    failures.append(
                        {"dataset": dataset, "run": run_dir.name, **result}
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                failures.append(
                    {"dataset": dataset, "run": run_dir.name, "error": str(exc)}
                )
    write_csv(audit_root / "engine_audit.csv", rows)
    write_json(
        audit_root / "audit_manifest.json",
        {
            "audit_id": args.audit_id,
            "datasets": args.dataset,
            "failures": failures,
            "notes": [
                "Saved engines were inspected without rebuilding or inference.",
                "Ultralytics JSON prefixes were separated before TensorRT deserialization.",
                "Q/DQ layer names verify an INT8 graph but do not prove per-channel weights.",
            ],
        },
    )
    print(json.dumps({"audit_root": str(audit_root), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
