from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_statistics import image_cluster_bootstrap, logistic_adjustment
from dataset import dataset_statistics, load_dataset, load_dataset_config
from diagnostics import (
    SCALES,
    confidence_bin_edges,
    decompose_failure_modes,
    false_positive_counts,
    pair_predictions,
    paired_gt_transitions,
    stratified_qfr,
    summarize_transitions,
)
from evaluation import evaluate_official_coco

PAPER_METRICS = ("AP", "AP_S", "AP_M", "AP_L")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_official_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("Cannot aggregate an empty official-metrics list")
    return {
        key: statistics.mean(float(item[key]) for item in metrics)
        for key in PAPER_METRICS
    }


def degradation_row(
    dataset: str,
    setting: str,
    fp32: dict[str, Any],
    candidates: list[dict[str, Any]],
    sources: list[Path],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Summarize official-COCO AP degradation without using exploratory metrics."""
    if len(candidates) != len(sources):
        raise ValueError("Each candidate metric must have exactly one source artifact")
    mean_metrics = mean_official_metrics(candidates)
    row: dict[str, Any] = {
        "dataset": dataset,
        "setting": setting,
        "seed_n": len(candidates),
    }
    for suffix, metric in (("AP", "AP"), ("S", "AP_S"), ("M", "AP_M"), ("L", "AP_L")):
        baseline = float(fp32[metric])
        candidate = mean_metrics[metric]
        drop = baseline - candidate
        row[f"FP32_{metric}"] = baseline
        row[f"INT8_{metric}"] = candidate
        row[f"D_{suffix}"] = drop
        row[f"RD_{suffix}"] = drop / baseline if abs(baseline) > tolerance else None
    row["SSG"] = row["D_S"] - row["D_L"]
    row["SSR"] = row["D_S"] / row["D_L"] if abs(row["D_L"]) > tolerance else None
    row["warning"] = (
        "SSR undefined because official D_L is near zero" if row["SSR"] is None else ""
    )
    row["source"] = ";".join(str(path) for path in sources)
    return row


def write_grouped_bar(
    rows: list[dict[str, Any]],
    label_key: str,
    value_keys: list[str],
    target: Path,
    ylabel: str,
    title: str,
) -> None:
    """Write a compact figure and its exact source rows side by side."""
    if not rows:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [label_key, *value_keys, "source"]
    write_csv(
        target.with_suffix(".csv"),
        [{field: row.get(field) for field in fields} for row in rows],
    )
    x = np.arange(len(rows))
    width = 0.8 / max(1, len(value_keys))
    figure, axis = plt.subplots(figsize=(max(6, len(rows) * 1.25), 4.5))
    for index, key in enumerate(value_keys):
        values = [np.nan if row.get(key) is None else row[key] for row in rows]
        axis.bar(
            x + (index - (len(value_keys) - 1) / 2) * width,
            values,
            width,
            label=key,
        )
    axis.set_xticks(x, [str(row[label_key]) for row in rows], rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_controlled_pair(
    fp_dir: Path, candidate_dir: Path, fp_ids: list[Any]
) -> None:
    fp_info = load_json(fp_dir / "run_info.json")
    candidate_info = load_json(candidate_dir / "run_info.json")
    if fp_info.get("checkpoint") != candidate_info.get("checkpoint"):
        raise ValueError("checkpoint differs from FP32")
    if load_json(candidate_dir / "evaluation_image_ids.json") != fp_ids:
        raise ValueError("evaluation image IDs differ from FP32")
    fp_config = yaml.safe_load(
        (fp_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    candidate_config = yaml.safe_load(
        (candidate_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    for key in ("model", "evaluation", "scale_bins"):
        if fp_config["global"].get(key) != candidate_config["global"].get(key):
            raise ValueError(f"controlled global setting differs from FP32: {key}")
    for key in ("name", "format", "val"):
        if fp_config["dataset"].get(key) != candidate_config["dataset"].get(key):
            raise ValueError(f"controlled dataset setting differs from FP32: {key}")


def flatten_official(
    dataset: str, run: str, metrics: dict[str, Any], source: Path
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "run": run,
        "AP": metrics["AP"],
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "AP_S": metrics["AP_S"],
        "AP_M": metrics["AP_M"],
        "AP_L": metrics["AP_L"],
    }
    for scale, short in (("small", "S"), ("medium", "M"), ("large", "L")):
        row[f"AP50_{short}"] = metrics["scale_iou"][scale]["AP50"]
        row[f"AP75_{short}"] = metrics["scale_iou"][scale]["AP75"]
    row["source"] = str(source)
    return row


def distribution_rows(run: str, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    fields = ("d_raw", "d_norm", "delta_w_relative", "delta_h_relative", "delta_score")
    for scale in SCALES:
        selected = [value for value in values if value["scale"] == scale]
        for field in fields:
            data = np.asarray(
                [
                    float(value[field])
                    for value in selected
                    if value.get(field) is not None
                ]
            )
            rows.append(
                {
                    "run": run,
                    "scale": scale,
                    "metric": field,
                    "n": int(data.size),
                    "median": float(np.median(data)) if data.size else None,
                    "q1": float(np.percentile(data, 25)) if data.size else None,
                    "q3": float(np.percentile(data, 75)) if data.size else None,
                    "negative_fraction": float(np.mean(data < 0))
                    if data.size
                    else None,
                    "p05": float(np.percentile(data, 5)) if data.size else None,
                    "p95": float(np.percentile(data, 95)) if data.size else None,
                }
            )
    return rows


def transition_shift_rows(run: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scale in SCALES:
        selected = [
            row for row in rows if row["scale"] == scale and row["transition"] == "11"
        ]
        metrics = {
            "delta_score": [
                float(row["int8_score"]) - float(row["fp32_score"]) for row in selected
            ],
            "delta_iou": [
                float(row["int8_iou"]) - float(row["fp32_iou"]) for row in selected
            ],
        }
        for metric, values in metrics.items():
            data = np.asarray(values)
            output.append(
                {
                    "run": run,
                    "scale": scale,
                    "metric": metric,
                    "n": int(data.size),
                    "median": float(np.median(data)) if data.size else None,
                    "q1": float(np.percentile(data, 25)) if data.size else None,
                    "q3": float(np.percentile(data, 75)) if data.size else None,
                    "negative_fraction": float(np.mean(data < 0))
                    if data.size
                    else None,
                    "p05": float(np.percentile(data, 5)) if data.size else None,
                    "p95": float(np.percentile(data, 95)) if data.size else None,
                }
            )
    return output


def transition_summary_rows(
    run: str, score: float, iou: float, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "run": run,
            "score_threshold": score,
            "iou_threshold": iou,
            "scale": scale,
            **summary[scale],
        }
        for scale in SCALES
    ]


def ap_decomposition_rows(
    run: str,
    fp32: dict[str, Any],
    int8: dict[str, Any],
    restored: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for metric in ("AP", "AP_S", "AP_M", "AP_L"):
        fp_value = float(fp32[metric])
        int8_value = float(int8[metric])
        restored_value = float(restored[metric])
        rows.append(
            {
                "run": run,
                "metric": metric,
                "fp32": fp_value,
                "int8": int8_value,
                "score_restored_int8": restored_value,
                "total_drop": fp_value - int8_value,
                "geometry_and_detection_set_component": fp_value - restored_value,
                "ranking_component": restored_value - int8_value,
                "identity_residual": (fp_value - int8_value)
                - ((fp_value - restored_value) + (restored_value - int8_value)),
                "interpretation": "order-dependent score-restored counterfactual",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified frozen-protocol offline analysis"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--protocol", default="protocols/framing.yaml")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--include-calibration-ablation", action="store_true")
    parser.add_argument(
        "--reuse-analysis",
        help="Reuse completed per-run artifacts from an interrupted analysis with the identical protocol",
    )
    args = parser.parse_args()
    if (
        not args.analysis_id.strip()
        or "/" in args.analysis_id
        or ".." in args.analysis_id
    ):
        raise ValueError("--analysis-id must be a non-empty directory-safe name")
    config_path, protocol_path = (
        Path(args.config).resolve(),
        Path(args.protocol).resolve(),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol_document = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol = protocol_document["protocol"]
    scale_cfg = protocol["scale_bins"]
    if scale_cfg != config["scale_bins"]:
        raise ValueError("Frozen protocol scale bins differ from experiment config")
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    analysis_root = output_root / "analysis" / args.analysis_id
    if analysis_root.exists():
        raise FileExistsError(
            f"Analysis exists and will not be overwritten: {analysis_root}"
        )
    analysis_root.mkdir(parents=True)
    shutil.copy2(config_path, analysis_root / "config_resolved.yaml")
    shutil.copy2(protocol_path, analysis_root / "protocol_resolved.yaml")
    reuse_root = Path(args.reuse_analysis).resolve() if args.reuse_analysis else None
    reused_input_hashes: set[str] = set()
    if reuse_root is not None:
        reused_protocol = reuse_root / "protocol_resolved.yaml"
        if (
            not reused_protocol.is_file()
            or reused_protocol.read_bytes() != protocol_path.read_bytes()
        ):
            raise ValueError(
                "Reused analysis has a missing or different frozen protocol"
            )
        reused_manifest = load_json(reuse_root / "analysis_manifest.json")
        reused_input_hashes = {
            str(item["sha256"])
            for item in reused_manifest.get("inputs", [])
            if isinstance(item, dict) and item.get("sha256")
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config_path.parent,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        analysis_root / "analysis_resolved.json",
        {
            "arguments": vars(args),
            "config": str(config_path),
            "protocol": str(protocol_path),
            "output_root": str(output_root),
            "git_commit": commit or None,
        },
    )
    manifest: dict[str, Any] = {
        "analysis_id": args.analysis_id,
        "analysis_schema": "framing",
        "entry_point": "scripts/analyze.py",
        "datasets_requested": args.dataset,
        "protocol_sha256": sha256(protocol_path),
        "inputs": [],
        "failures": [],
        "notes": [
            "Inference was not rerun.",
            "Main AP metrics use official pycocotools COCOeval.",
            "RD, SSR and SSG summaries are derived only from official COCOeval AP.",
            "Loaded GT controls crowd/ignore handling; converted YOLO VisDrone labels do not retain original ignored regions.",
        ],
    }
    if reuse_root is not None:
        manifest["notes"].append(
            f"Completed per-run artifacts were reused from interrupted analysis {reuse_root} after exact protocol comparison."
        )
    official_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    degradation_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for dataset_path in args.dataset:
        try:
            dataset_cfg = load_dataset_config(dataset_path)
            dataset_config_path = Path(dataset_path).resolve()
            dataset_config_sha256 = sha256(dataset_config_path)
            manifest["inputs"].append(
                {
                    "path": str(dataset_config_path),
                    "sha256": dataset_config_sha256,
                }
            )
            ground_truth = load_dataset(dataset_cfg, "val")
            name = dataset_cfg["name"]
            visdrone_ignore_missing = (
                dataset_cfg.get("native_evaluator") == "visdrone"
                and dataset_cfg.get("format") != "visdrone"
            )
            if visdrone_ignore_missing:
                manifest["notes"].append(
                    f"{name}: converted labels omit VisDrone ignored regions; official AP and FP accounting are provisional and not paper-valid until original annotations are configured."
                )
            dataset_output = analysis_root / name
            dataset_output.mkdir()
            statistics_row = dataset_statistics(ground_truth, scale_cfg)
            statistics_row.update({"dataset": name, "split": "val"})
            count_rows.append(statistics_row)
            write_json(dataset_output / "validation_statistics.json", statistics_row)
            dataset_root = output_root / name
            fp_dir = dataset_root / "fp32"
            fp_ids = load_json(fp_dir / "evaluation_image_ids.json")
            manifest["inputs"].append(
                {
                    "path": str(fp_dir / "evaluation_image_ids.json"),
                    "sha256": sha256(fp_dir / "evaluation_image_ids.json"),
                }
            )
            id_lookup = {str(record["image_id"]): record for record in ground_truth}
            ground_truth = [id_lookup[str(image_id)] for image_id in fp_ids]
            fp_path = fp_dir / "predictions.json"
            fp_predictions = load_json(fp_path)
            manifest["inputs"].append({"path": str(fp_path), "sha256": sha256(fp_path)})
            fp_official_path = dataset_output / "official_metrics_fp32.json"
            reuse_dataset_allowed = dataset_config_sha256 in reused_input_hashes
            reused_dataset = (
                reuse_root / name
                if reuse_root is not None and reuse_dataset_allowed
                else None
            )
            if reuse_root is not None and not reuse_dataset_allowed:
                manifest["notes"].append(
                    f"{name}: dataset configuration is new or changed; prior per-run analysis artifacts were not reused."
                )
            reused_fp = (
                reused_dataset / "official_metrics_fp32.json"
                if reused_dataset is not None
                else None
            )
            if reused_fp is not None and reused_fp.is_file():
                shutil.copy2(reused_fp, fp_official_path)
                fp_official = load_json(fp_official_path)
            else:
                fp_official = evaluate_official_coco(
                    dataset_cfg, ground_truth, fp_predictions
                )
                write_json(fp_official_path, fp_official)
            official_rows.append(
                flatten_official(name, "fp32", fp_official, fp_official_path)
            )
            control_dir = dataset_root / "trt_fp16"
            try:
                validate_controlled_pair(fp_dir, control_dir, fp_ids)
                control_prediction_path = control_dir / "predictions.json"
                control_predictions = load_json(control_prediction_path)
                manifest["inputs"].append(
                    {
                        "path": str(control_prediction_path),
                        "sha256": sha256(control_prediction_path),
                    }
                )
                control_metrics_path = dataset_output / "official_metrics_trt_fp16.json"
                reused_control = (
                    reused_dataset / "official_metrics_trt_fp16.json"
                    if reused_dataset is not None
                    else None
                )
                if reused_control is not None and reused_control.is_file():
                    shutil.copy2(reused_control, control_metrics_path)
                    control_metrics = load_json(control_metrics_path)
                else:
                    control_metrics = evaluate_official_coco(
                        dataset_cfg, ground_truth, control_predictions
                    )
                    write_json(control_metrics_path, control_metrics)
                official_rows.append(
                    flatten_official(
                        name, "trt_fp16", control_metrics, control_metrics_path
                    )
                )
                for metric in ("AP", "AP_S", "AP_M", "AP_L"):
                    fp_value, control_value = (
                        float(fp_official[metric]),
                        float(control_metrics[metric]),
                    )
                    control_rows.append(
                        {
                            "dataset": name,
                            "metric": metric,
                            "pytorch_fp32": fp_value,
                            "tensorrt_fp16": control_value,
                            "absolute_drop": fp_value - control_value,
                            "relative_drop": (fp_value - control_value) / fp_value
                            if fp_value
                            else None,
                            "source": str(control_metrics_path),
                        }
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                manifest["failures"].append(
                    {"dataset": name, "run": "trt_fp16", "error": str(exc)}
                )
            run_names = [
                f"int8_random_s{seed}" for seed in config["calibration"]["random_seeds"]
            ]
            if args.include_calibration_ablation:
                run_names += ["int8_small_rich", "int8_scale_balanced"]
            run_official: list[tuple[str, dict[str, Any], Path]] = []
            quant_official: list[tuple[str, dict[str, Any], Path]] = []
            primary_edges: list[float] | None = None
            for run_name in run_names:
                run_dir = dataset_root / run_name
                prediction_path = run_dir / "predictions.json"
                try:
                    validate_controlled_pair(fp_dir, run_dir, fp_ids)
                    predictions = load_json(prediction_path)
                    manifest["inputs"].append(
                        {
                            "path": str(prediction_path),
                            "sha256": sha256(prediction_path),
                        }
                    )
                    reused_run = (
                        reused_dataset / run_name
                        if reused_dataset is not None
                        else None
                    )
                    reused_official = (
                        reused_dataset / f"official_metrics_{run_name}.json"
                        if reused_dataset is not None
                        else None
                    )
                    required_reused = (
                        "ap_decomposition.csv",
                        "bootstrap_ci.json",
                        "paired_transitions.json",
                        "prediction_pairs.json",
                        "regression.json",
                        "stratified_qfr.csv",
                        "threshold_sweep.csv",
                        "transition_summary.json",
                    )
                    if (
                        reused_run is not None
                        and reused_run.is_dir()
                        and reused_official is not None
                        and reused_official.is_file()
                        and all(
                            (reused_run / item).is_file() for item in required_reused
                        )
                    ):
                        run_output = dataset_output / run_name
                        shutil.copytree(reused_run, run_output)
                        official_path = (
                            dataset_output / f"official_metrics_{run_name}.json"
                        )
                        shutil.copy2(reused_official, official_path)
                        official = load_json(official_path)
                        official_rows.append(
                            flatten_official(name, run_name, official, official_path)
                        )
                        quant_official.append((run_name, official, official_path))
                        if run_name.startswith("int8_random_s"):
                            run_official.append((run_name, official, official_path))
                        reused_edges = reused_dataset / "confidence_bin_edges.json"
                        if (
                            reused_edges.is_file()
                            and not (
                                dataset_output / "confidence_bin_edges.json"
                            ).exists()
                        ):
                            shutil.copy2(
                                reused_edges,
                                dataset_output / "confidence_bin_edges.json",
                            )
                        continue
                    official = evaluate_official_coco(
                        dataset_cfg, ground_truth, predictions
                    )
                    official_path = dataset_output / f"official_metrics_{run_name}.json"
                    write_json(official_path, official)
                    official_rows.append(
                        flatten_official(name, run_name, official, official_path)
                    )
                    quant_official.append((run_name, official, official_path))
                    if run_name.startswith("int8_random_s"):
                        run_official.append((run_name, official, official_path))
                    sweep_rows: list[dict[str, Any]] = []
                    primary_rows: list[dict[str, Any]] | None = None
                    for iou in protocol["iou_thresholds"]:
                        for score in protocol["score_thresholds"]:
                            transitions, _ = paired_gt_transitions(
                                name,
                                ground_truth,
                                fp_predictions,
                                predictions,
                                scale_cfg,
                                float(score),
                                float(iou),
                            )
                            sweep_rows.extend(
                                transition_summary_rows(
                                    run_name,
                                    float(score),
                                    float(iou),
                                    summarize_transitions(transitions),
                                )
                            )
                            if float(iou) == float(protocol["primary_iou"]) and float(
                                score
                            ) == float(protocol["primary_score"]):
                                primary_rows = transitions
                    if primary_rows is None:
                        raise ValueError(
                            "Primary score/IoU is absent from frozen sweep"
                        )
                    run_output = dataset_output / run_name
                    run_output.mkdir()
                    write_csv(run_output / "threshold_sweep.csv", sweep_rows)
                    write_json(run_output / "paired_transitions.json", primary_rows)
                    summary = summarize_transitions(primary_rows)
                    write_json(run_output / "transition_summary.json", summary)
                    write_csv(
                        run_output / "continuous_shift_summary.csv",
                        transition_shift_rows(run_name, primary_rows),
                    )
                    failure_modes = decompose_failure_modes(
                        primary_rows,
                        predictions,
                        float(protocol["primary_score"]),
                        float(protocol["primary_iou"]),
                        float(protocol["failure_modes"]["minimum_localization_iou"]),
                    )
                    write_json(run_output / "failure_modes.json", failure_modes)
                    write_csv(
                        run_output / "failure_mode_summary.csv",
                        [
                            {
                                "scale": scale,
                                "mode": mode,
                                "count": sum(
                                    row["scale"] == scale and row["mode"] == mode
                                    for row in failure_modes
                                ),
                            }
                            for scale in SCALES
                            for mode in protocol["failure_modes"]["priority"]
                        ],
                    )
                    if primary_edges is None:
                        primary_edges = confidence_bin_edges(
                            primary_rows, int(protocol["stratification"]["bins"])
                        )
                        write_json(
                            dataset_output / "confidence_bin_edges.json", primary_edges
                        )
                    write_csv(
                        run_output / "stratified_qfr.csv",
                        stratified_qfr(primary_rows, primary_edges),
                    )
                    model = logistic_adjustment(
                        primary_rows,
                        int(protocol["regression"]["maximum_iterations"]),
                        float(protocol["regression"]["tolerance"]),
                    )
                    write_json(run_output / "regression.json", model)
                    replicates = (
                        args.bootstrap_replicates
                        if args.bootstrap_replicates is not None
                        else int(protocol["bootstrap"]["replicates"])
                    )
                    bootstrap = image_cluster_bootstrap(
                        primary_rows,
                        replicates,
                        int(protocol["bootstrap"]["seed"]),
                        int(protocol["regression"]["maximum_iterations"]),
                        float(protocol["regression"]["tolerance"]),
                    )
                    write_json(run_output / "bootstrap_ci.json", bootstrap)
                    if visdrone_ignore_missing:
                        write_json(
                            run_output / "false_positive_counts_skipped.json",
                            {
                                "status": "skipped",
                                "reason": "Converted VisDrone labels omit original ignored regions",
                            },
                        )
                    else:
                        fps = false_positive_counts(
                            ground_truth,
                            fp_predictions,
                            predictions,
                            scale_cfg,
                            float(protocol["primary_score"]),
                            float(protocol["primary_iou"]),
                        )
                        write_csv(run_output / "false_positive_counts.csv", fps)
                    pairs = pair_predictions(
                        fp_predictions,
                        predictions,
                        ground_truth,
                        scale_cfg,
                        float(protocol["prediction_pairing"]["score_threshold"]),
                        float(protocol["prediction_pairing"]["minimum_iou"]),
                    )
                    write_json(run_output / "prediction_pairs.json", pairs)
                    write_csv(
                        run_output / "displacement_summary.csv",
                        distribution_rows(run_name, pairs),
                    )
                    decomposition_pairs = pair_predictions(
                        fp_predictions,
                        predictions,
                        ground_truth,
                        scale_cfg,
                        float(protocol["ap_decomposition"]["pairing_score_threshold"]),
                        float(protocol["ap_decomposition"]["minimum_pair_iou"]),
                    )
                    restored_predictions = [
                        dict(prediction) for prediction in predictions
                    ]
                    for pair in decomposition_pairs:
                        restored_predictions[pair["int8_prediction_id"]]["score"] = (
                            pair["fp32_score"]
                        )
                    restored_metrics = evaluate_official_coco(
                        dataset_cfg, ground_truth, restored_predictions
                    )
                    write_json(
                        run_output / "score_restored_official_metrics.json",
                        restored_metrics,
                    )
                    write_csv(
                        run_output / "ap_decomposition.csv",
                        ap_decomposition_rows(
                            run_name, fp_official, official, restored_metrics
                        ),
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    manifest["failures"].append(
                        {"dataset": name, "run": run_name, "error": str(exc)}
                    )
            if run_official:
                for key in ("AP", "AP_S", "AP_M", "AP_L"):
                    values = [float(metrics[key]) for _, metrics, _ in run_official]
                    fp_value = float(fp_official[key])
                    drop_values = [fp_value - value for value in values]
                    paper_rows.append(
                        {
                            "dataset": name,
                            "metric": key,
                            "fp32": fp_value,
                            "int8_mean": statistics.mean(values),
                            "int8_std": statistics.pstdev(values)
                            if len(values) > 1
                            else 0.0,
                            "delta_ap_mean": statistics.mean(drop_values),
                            "delta_ap_std": statistics.pstdev(drop_values)
                            if len(values) > 1
                            else 0.0,
                            "relative_drop": statistics.mean(drop_values) / fp_value
                            if fp_value
                            else None,
                            "seed_n": len(values),
                            "source": str(dataset_output),
                        }
                    )
                random_summary = degradation_row(
                    name,
                    "int8_random_seed_mean",
                    fp_official,
                    [metrics for _, metrics, _ in run_official],
                    [path for _, _, path in run_official],
                )
                degradation_rows.append(random_summary)
                dataset_calibration_rows = [random_summary]
                for run_name, metrics, source in quant_official:
                    if run_name.startswith("int8_random_s"):
                        continue
                    dataset_calibration_rows.append(
                        degradation_row(
                            name, run_name, fp_official, [metrics], [source]
                        )
                    )
                if len(dataset_calibration_rows) > 1:
                    calibration_rows.extend(dataset_calibration_rows)

                seed_plot_rows = [
                    {
                        "run": run_name,
                        "AP_S": float(metrics["AP_S"]),
                        "AP_M": float(metrics["AP_M"]),
                        "AP_L": float(metrics["AP_L"]),
                        "source": str(source),
                    }
                    for run_name, metrics, source in run_official
                ]
                write_grouped_bar(
                    seed_plot_rows,
                    "run",
                    ["AP_S", "AP_M", "AP_L"],
                    dataset_output / "figures" / "seed_variance.png",
                    "Official COCO AP",
                    f"{name}: random calibration seeds",
                )
                scale_rows = [
                    {
                        "scale": scale,
                        "D": random_summary[f"D_{short}"],
                        "RD": random_summary[f"RD_{short}"],
                        "source": random_summary["source"],
                    }
                    for scale, short in (
                        ("small", "S"),
                        ("medium", "M"),
                        ("large", "L"),
                    )
                ]
                write_grouped_bar(
                    scale_rows,
                    "scale",
                    ["D"],
                    dataset_output / "figures" / "scale_degradation.png",
                    "Official COCO AP drop",
                    f"{name}: scale degradation",
                )
                write_grouped_bar(
                    scale_rows,
                    "scale",
                    ["RD"],
                    dataset_output / "figures" / "relative_degradation.png",
                    "Relative official COCO AP drop",
                    f"{name}: relative scale degradation",
                )
                if len(dataset_calibration_rows) > 1:
                    write_grouped_bar(
                        dataset_calibration_rows,
                        "setting",
                        ["INT8_AP_S", "INT8_AP_M", "INT8_AP_L"],
                        dataset_output / "figures" / "calibration_comparison.png",
                        "Official COCO AP",
                        f"{name}: calibration comparison",
                    )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            manifest["failures"].append(
                {"dataset": dataset_path, "run": "dataset", "error": str(exc)}
            )
    write_csv(analysis_root / "official_metrics.csv", official_rows)
    write_csv(analysis_root / "paper_ap_table.csv", paper_rows)
    write_csv(analysis_root / "pipeline_controls.csv", control_rows)
    write_csv(analysis_root / "validation_statistics.csv", count_rows)
    write_csv(analysis_root / "official_degradation_summary.csv", degradation_rows)
    write_csv(analysis_root / "calibration_comparison.csv", calibration_rows)
    write_grouped_bar(
        degradation_rows,
        "dataset",
        ["RD_S", "RD_M", "RD_L"],
        analysis_root / "figures" / "cross_dataset_relative_degradation.png",
        "Relative official COCO AP drop",
        "Cross-dataset relative degradation",
    )
    write_grouped_bar(
        degradation_rows,
        "dataset",
        ["SSR"],
        analysis_root / "figures" / "cross_dataset_ssr.png",
        "Small-to-large sensitivity ratio",
        "Cross-dataset scale sensitivity",
    )
    write_json(analysis_root / "analysis_manifest.json", manifest)
    print(
        json.dumps(
            {"analysis_root": str(analysis_root), "failures": manifest["failures"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
