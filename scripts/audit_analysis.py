#!/usr/bin/env python3
"""Audit frozen analysis artifacts for paper-facing validity and diagnostics.

Every deliverable here is computed offline from the frozen analysis artifacts.
No inference, no engine build, and no evaluation of new predictions is run, and
the frozen protocol is never modified: post-hoc variants are labelled as such.

Items closed
------------
A1  Stratification coverage diagnosis: FP32 score distribution per scale versus
    the pooled decile edges, plus post-hoc coarser-binning and common-support
    variants so an option can be chosen on evidence.
A2  FP32 baseline provenance: byte-level determinism, native-versus-pycocotools
    agreement, cross-analysis comparison, and the total-drop reconciliation.
A3  Score shift of surviving detections (transition 11) aggregated over seeds.
A4  Score shift of the lost detections (transition 10) classified as confidence
    failures -- deep collapse versus threshold grazing -- and the full
    5 score x 2 IoU threshold sweep ordering.
B1  Pipeline validation summary from the FP16 control and the engine audit.
B2  Ground-truth instance counts per scale with every share reported.
B3  False-positive accounting per predicted-box scale bin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCALES = ("small", "medium", "large")
SHORT = {"small": "S", "medium": "M", "large": "L"}
SEED_RUNS = tuple(f"int8_random_s{seed}" for seed in range(5))
POST_HOC_BIN_COUNTS = (10, 5, 4, 3)


# --------------------------------------------------------------------------- io


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {key: None for key in ("min", "p05", "q1", "median", "q3", "p95", "max")}
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "q1": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q3": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values)


# ------------------------------------------------------------------- A1 support


def fp32_success_scores(transitions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """FP32-matched scores per scale; identical for every seed of a dataset."""
    by_scale: dict[str, list[float]] = defaultdict(list)
    for row in transitions:
        if row["fp32_detected"]:
            by_scale[row["scale"]].append(float(row["fp32_score"]))
    return {
        scale: np.sort(np.asarray(by_scale.get(scale, []), dtype=float))
        for scale in SCALES
    }


def bin_counts(scores: np.ndarray, edges: list[float]) -> list[int]:
    """Count with the frozen half-open convention and a closed final bin."""
    counts = []
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            counts.append(int(np.count_nonzero((scores >= left) & (scores <= right))))
        else:
            counts.append(int(np.count_nonzero((scores >= left) & (scores < right))))
    return counts


def pooled_edges(per_scale: dict[str, np.ndarray], bins: int) -> list[float]:
    pooled = np.concatenate([per_scale[scale] for scale in SCALES])
    return [
        float(value) for value in np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1))
    ]


def a1_histogram(
    dataset: str, per_scale: dict[str, np.ndarray], edges: list[float], minimum: int
) -> list[dict[str, Any]]:
    counts = {scale: bin_counts(per_scale[scale], edges) for scale in SCALES}
    rows = []
    for index in range(len(edges) - 1):
        usable = all(counts[scale][index] >= minimum for scale in SCALES)
        for scale in SCALES:
            rows.append(
                {
                    "dataset": dataset,
                    "bin": index + 1,
                    "left": edges[index],
                    "right": edges[index + 1],
                    "scale": scale,
                    "n": counts[scale][index],
                    "share_of_scale": counts[scale][index] / per_scale[scale].size
                    if per_scale[scale].size
                    else None,
                    "meets_minimum": counts[scale][index] >= minimum,
                    "bin_usable_for_all_scales": usable,
                }
            )
    return rows


def a1_distribution(
    dataset: str, per_scale: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "scale": scale,
            "n_fp32_matched": int(per_scale[scale].size),
            **quantiles(per_scale[scale]),
        }
        for scale in SCALES
    ]


def required_common_bins(bins: int, frozen_bins: int, frozen_required: int) -> int:
    """Carry the frozen 8-of-10 coverage rule across to coarser bin counts."""
    return math.ceil(bins * frozen_required / frozen_bins)


def a1_binning_variants(
    dataset: str,
    per_scale: dict[str, np.ndarray],
    minimum: int,
    frozen_bins: int,
    frozen_required: int,
) -> list[dict[str, Any]]:
    """Frozen decile plus post-hoc coarser binning, evaluated identically."""
    rows = []
    for bins in POST_HOC_BIN_COUNTS:
        edges = pooled_edges(per_scale, bins)
        counts = {scale: bin_counts(per_scale[scale], edges) for scale in SCALES}
        usable = sum(
            all(counts[scale][index] >= minimum for scale in SCALES)
            for index in range(bins)
        )
        required = (
            frozen_required
            if bins == frozen_bins
            else required_common_bins(bins, frozen_bins, frozen_required)
        )
        rows.append(
            {
                "dataset": dataset,
                "bins": bins,
                "status": "frozen_protocol"
                if bins == frozen_bins
                else "post_hoc_variant",
                "valid_common_bins": usable,
                "required_for_decision": required,
                "coverage_available": usable >= required,
                "minimum_per_scale": minimum,
                "smallest_scale_cell": min(min(counts[scale]) for scale in SCALES),
            }
        )
    return rows


def a1_common_support(
    dataset: str, per_scale: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Option-C evidence: the score band every scale actually occupies."""
    low = max(float(per_scale[scale].min()) for scale in SCALES)
    high = min(float(per_scale[scale].max()) for scale in SCALES)
    rows = []
    for scale in SCALES:
        scores = per_scale[scale]
        inside = int(np.count_nonzero((scores >= low) & (scores <= high)))
        rows.append(
            {
                "dataset": dataset,
                "scale": scale,
                "common_support_low": low,
                "common_support_high": high,
                "n_total": int(scores.size),
                "n_inside_common_support": inside,
                "fraction_inside": inside / scores.size if scores.size else None,
                "median_inside": float(
                    np.median(scores[(scores >= low) & (scores <= high)])
                )
                if inside
                else None,
            }
        )
    return rows


def a1_top_bin_concentration(
    per_scale: dict[str, np.ndarray], edges: list[float]
) -> dict[str, float | None]:
    """Share of each scale sitting in the top two pooled deciles."""
    cut = edges[-3]
    return {
        scale: float(np.count_nonzero(per_scale[scale] >= cut) / per_scale[scale].size)
        if per_scale[scale].size
        else None
        for scale in SCALES
    }


# ------------------------------------------------------------------- A2 support


def a2_baseline_evidence(
    output_root: Path,
    analysis_root: Path,
    comparison_root: Path | None,
    datasets: list[str],
) -> dict[str, Any]:
    per_dataset = {}
    for dataset in datasets:
        primary = output_root / dataset / "fp32" / "predictions.json"
        repro_candidates = sorted(
            (output_root / dataset).glob("fp32_repro*/predictions.json")
        )
        repro = repro_candidates[0] if len(repro_candidates) == 1 else None
        record: dict[str, Any] = {
            "fp32_predictions_sha256": sha256(primary),
            "fp32_repro_predictions_sha256": sha256(repro) if repro else None,
            "fp32_repro_prediction_path": str(repro) if repro else None,
            "fp32_repro_candidate_count": len(repro_candidates),
        }
        record["inference_bit_identical"] = (
            record["fp32_repro_predictions_sha256"] == record["fp32_predictions_sha256"]
        )
        official = read_json(analysis_root / dataset / "official_metrics_fp32.json")
        native = read_json(output_root / dataset / "fp32" / "metrics.json")["native"]
        record["run_local_native_AP"] = optional_float(native.get("AP"))
        record["analysis_pycocotools_AP"] = float(official["AP"])
        record["native_minus_pycocotools_AP"] = (
            record["run_local_native_AP"] - record["analysis_pycocotools_AP"]
            if record["run_local_native_AP"] is not None
            else None
        )
        record["run_local_reports_scale_ap"] = native.get("AP_S") is not None
        record["scale_ap_source"] = "analysis_pycocotools_only"
        per_dataset[dataset] = record
    evidence: dict[str, Any] = {"per_dataset": per_dataset}

    if comparison_root is not None:
        current = {
            (row["dataset"], row["metric"]): row
            for row in read_csv(analysis_root / "paper_ap_table.csv")
        }
        previous = {
            (row["dataset"], row["metric"]): row
            for row in read_csv(comparison_root / "paper_ap_table.csv")
        }
        rows = []
        for key in sorted(set(current) & set(previous)):
            dataset, metric = key
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "fp32_current": float(current[key]["fp32"]),
                    "fp32_previous": float(previous[key]["fp32"]),
                    "fp32_difference": float(current[key]["fp32"])
                    - float(previous[key]["fp32"]),
                    "delta_ap_current": float(current[key]["delta_ap_mean"]),
                    "delta_ap_previous": float(previous[key]["delta_ap_mean"]),
                    "delta_ap_difference": float(current[key]["delta_ap_mean"])
                    - float(previous[key]["delta_ap_mean"]),
                }
            )
        evidence["cross_analysis_comparison"] = {
            "current_analysis": str(analysis_root),
            "previous_analysis": str(comparison_root),
            "rows": rows,
            "datasets_with_any_fp32_change": sorted(
                {row["dataset"] for row in rows if abs(row["fp32_difference"]) > 1e-12}
            ),
        }

    ap_table = {
        row["dataset"]: row
        for row in read_csv(analysis_root / "paper_ap_table.csv")
        if row["metric"] == "AP"
    }
    reconciliation = []
    for dataset in datasets:
        decomposition = [
            row
            for run in SEED_RUNS
            for row in read_csv(analysis_root / dataset / run / "ap_decomposition.csv")
            if row["metric"] == "AP"
        ]
        mean_total = statistics.mean(float(row["total_drop"]) for row in decomposition)
        table_value = float(ap_table[dataset]["delta_ap_mean"])
        reconciliation.append(
            {
                "dataset": dataset,
                "paper_ap_table_delta_ap_mean": table_value,
                "ap_decomposition_total_drop_mean": mean_total,
                "difference": table_value - mean_total,
                "agrees_within_1e_9": abs(table_value - mean_total) < 1e-9,
            }
        )
    evidence["total_drop_reconciliation"] = reconciliation
    return evidence


# -------------------------------------------------- regression interpretability


def design_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Leverage and support checks for the covariates the model actually uses.

    Mirrors ``analysis_statistics._design`` so the numbers describe the fitted model
    rather than a re-specified one.
    """
    selected = [row for row in rows if row["fp32_detected"]]
    scores = np.asarray([float(row["fp32_score"]) for row in selected])
    ious = np.asarray([float(row["fp32_iou"]) for row in selected])

    def standardized(values: np.ndarray) -> np.ndarray:
        std = float(values.std())
        return (values - values.mean()) / std if std > 0 else values * 0.0

    design = np.column_stack(
        [
            [row["scale"] == "medium" for row in selected],
            [row["scale"] == "large" for row in selected],
            standardized(scores),
            standardized(ious),
        ]
    ).astype(float)
    names = ("scale_medium", "scale_large", "fp32_score_z", "fp32_iou_z")
    variance_inflation = {}
    for position, name in enumerate(names):
        target = design[:, position]
        others = np.column_stack(
            [np.ones(len(target)), np.delete(design, position, axis=1)]
        )
        total = float(target.var())
        if total == 0.0:
            # A constant column carries no variance to inflate, which happens
            # when a scale is absent from the split.
            variance_inflation[name] = None
            continue
        coefficients, *_ = np.linalg.lstsq(others, target, rcond=None)
        residual = target - others @ coefficients
        explained = 1.0 - float(residual.var()) / total
        variance_inflation[name] = (
            float(1.0 / (1.0 - explained)) if explained < 1.0 else None
        )
    small = np.asarray(
        [float(row["fp32_score"]) for row in selected if row["scale"] == "small"]
    )
    reference_ceiling = float(np.percentile(small, 95))
    support = {}
    events = {}
    for scale in SCALES:
        values = np.asarray(
            [float(row["fp32_score"]) for row in selected if row["scale"] == scale]
        )
        support[scale] = (
            float(np.mean(values <= reference_ceiling)) if values.size else None
        )
        events[scale] = sum(
            1 for row in selected if row["scale"] == scale and row["transition"] == "10"
        )
    return {
        "n": len(selected),
        "variance_inflation": variance_inflation,
        "reference_score_ceiling_p95_small": reference_ceiling,
        "fraction_at_or_below_reference_ceiling": support,
        "flip_events": events,
    }


def regression_interpretability(
    analysis_root: Path,
    datasets: list[str],
    minimum_events: int,
    minimum_support: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-seed coefficients with the checks that decide interpretability.

    A seed-averaged coefficient is only meaningful when the per-seed intervals
    agree in sign. Where they point in opposite directions the mean describes no
    single quantity, and that has to be said rather than averaged away.
    """
    per_seed: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    for dataset in datasets:
        diagnostics = design_diagnostics(
            read_json(
                analysis_root / dataset / SEED_RUNS[0] / "paired_transitions.json"
            )
        )
        for term, scale in (("scale_medium", "medium"), ("scale_large", "large")):
            signs: list[str] = []
            values: list[float] = []
            for run in SEED_RUNS:
                model = read_json(analysis_root / dataset / run / "regression.json")
                bootstrap = read_json(
                    analysis_root / dataset / run / "bootstrap_ci.json"
                )
                coefficient = float(model["coefficients"][term]["log_odds"])
                ci = bootstrap["regression_log_odds"][term]
                lower, upper = float(ci["lower_95"]), float(ci["upper_95"])
                sign = (
                    "positive"
                    if lower > 0
                    else ("negative" if upper < 0 else "spans_zero")
                )
                signs.append(sign)
                values.append(coefficient)
                per_seed.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "term": term,
                        "reference_scale": "small",
                        "log_odds": coefficient,
                        "odds_ratio": math.exp(coefficient),
                        "bootstrap_lower_95": lower,
                        "bootstrap_median": float(ci["median"]),
                        "bootstrap_upper_95": upper,
                        "ci_sign": sign,
                    }
                )
            conflicting = "positive" in signs and "negative" in signs
            events = diagnostics["flip_events"][scale]
            support = diagnostics["fraction_at_or_below_reference_ceiling"][scale]
            vif = diagnostics["variance_inflation"][term]
            reasons = []
            if conflicting:
                reasons.append("per_seed_bootstrap_intervals_disagree_in_sign")
            if events < minimum_events:
                reasons.append(f"fewer_than_{minimum_events}_flip_events")
            if support is not None and support < minimum_support:
                reasons.append("weak_confidence_overlap_with_reference_scale")
            mean_value, std_value = mean_std(values)
            verdicts.append(
                {
                    "dataset": dataset,
                    "term": term,
                    "log_odds_mean": mean_value,
                    "log_odds_std": std_value,
                    "seed_signs": ";".join(signs),
                    "distinct_seed_signs": len(set(signs)),
                    "signs_conflict": conflicting,
                    "flip_events_identifying_term": events,
                    "fraction_at_or_below_reference_ceiling": support,
                    "variance_inflation_factor": vif,
                    "interpretable_as_seed_mean": not reasons,
                    "reasons_withheld": ";".join(reasons) if reasons else None,
                    "reporting_requirement": "report_per_seed_only"
                    if reasons
                    else "seed_mean_admissible",
                }
            )
    return per_seed, verdicts


# ------------------------------------------------- variance source attribution


def variance_decomposition(
    analysis_root: Path, datasets: list[str]
) -> list[dict[str, Any]]:
    """Split QFR spread into calibration-seed and evaluation-sampling parts.

    The image-clustered bootstrap already measures how much a single run's QFR
    would move under resampling of the evaluation set. Comparing that with the
    spread across calibration seeds says which source dominates, and therefore
    whether a calibration ablation could reduce the variance at all.
    """
    rows = []
    for dataset in datasets:
        for scale in SCALES:
            point: list[float] = []
            widths: list[float] = []
            for run in SEED_RUNS:
                summary = read_json(
                    analysis_root / dataset / run / "transition_summary.json"
                )[scale]
                bootstrap = read_json(
                    analysis_root / dataset / run / "bootstrap_ci.json"
                )
                qfr = summary["qfr"]
                interval = bootstrap["qfr"][scale]
                if qfr is None or interval["lower_95"] is None:
                    continue
                point.append(float(qfr))
                widths.append(float(interval["upper_95"]) - float(interval["lower_95"]))
            if len(point) < 2:
                continue
            between = statistics.pstdev(point)
            # A percentile interval spans about 3.92 standard deviations.
            within = statistics.mean(widths) / 3.92
            ratio = between / within if within > 0 else None
            if ratio is None:
                dominant = "undetermined"
            elif ratio >= 3.0:
                dominant = "calibration_seed"
            elif ratio <= 1.0:
                dominant = "evaluation_set_sampling"
            else:
                dominant = "comparable"
            rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "qfr_per_seed": ";".join(f"{value:.4f}" for value in point),
                    "between_seed_sd": between,
                    "within_seed_bootstrap_sd": within,
                    "ratio_between_over_within": ratio,
                    "dominant_variance_source": dominant,
                    "seed_std_understates_uncertainty": bool(
                        ratio is not None and ratio < 1.0
                    ),
                    "calibration_ablation_could_help": dominant == "calibration_seed",
                    "seed_n": len(point),
                }
            )
    return rows


def qfr_bootstrap_intervals(
    analysis_root: Path, datasets: list[str]
) -> list[dict[str, Any]]:
    """Image-clustered QFR intervals per run, plus the seed-pooled summary.

    The averaged interval on the pooled row measures evaluation-set resampling
    within a run and carries none of the spread across calibration seeds. Where
    that spread dominates, the averaged interval is far narrower than the real
    uncertainty, so every pooled row also carries the between-seed deviation and
    a widened interval that adds it back.
    """
    rows = []
    for dataset in datasets:
        for scale in SCALES:
            medians, lowers, uppers, points = [], [], [], []
            for run in SEED_RUNS:
                bootstrap = read_json(
                    analysis_root / dataset / run / "bootstrap_ci.json"
                )
                interval = bootstrap["qfr"][scale]
                summary = read_json(
                    analysis_root / dataset / run / "transition_summary.json"
                )[scale]
                rows.append(
                    {
                        "dataset": dataset,
                        "scale": scale,
                        "run": run,
                        "qfr_point": summary["qfr"],
                        "bootstrap_lower_95": interval["lower_95"],
                        "bootstrap_median": interval["median"],
                        "bootstrap_upper_95": interval["upper_95"],
                        "replicates": interval["n"],
                        "unit": "image",
                        "between_seed_sd": None,
                        "combined_lower_95": None,
                        "combined_upper_95": None,
                        "averaged_interval_excludes_between_seed_variance": False,
                        "combined_interval_clipped_to_unit_range": None,
                        "reporting_requirement": None,
                    }
                )
                medians.append(float(interval["median"]))
                lowers.append(float(interval["lower_95"]))
                uppers.append(float(interval["upper_95"]))
                points.append(float(summary["qfr"]))
            point = statistics.mean(points)
            between = statistics.pstdev(points)
            within = (
                statistics.mean(upper - lower for lower, upper in zip(lowers, uppers))
                / 3.92
            )
            combined = math.sqrt(between**2 + within**2)
            raw_lower, raw_upper = point - 1.96 * combined, point + 1.96 * combined
            # A rate cannot leave [0, 1]. When the normal approximation does, the
            # seed spread is too large for any single interval to describe, which
            # is itself the finding: report the seeds instead of a pooled number.
            degenerate = raw_lower < 0.0 or raw_upper > 1.0
            rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "run": "seed_mean",
                    "qfr_point": point,
                    "bootstrap_lower_95": statistics.mean(lowers),
                    "bootstrap_median": statistics.mean(medians),
                    "bootstrap_upper_95": statistics.mean(uppers),
                    "replicates": None,
                    "unit": "image_bootstrap_averaged_over_seeds",
                    "between_seed_sd": between,
                    "combined_lower_95": min(max(raw_lower, 0.0), 1.0),
                    "combined_upper_95": min(max(raw_upper, 0.0), 1.0),
                    "averaged_interval_excludes_between_seed_variance": True,
                    "combined_interval_clipped_to_unit_range": degenerate,
                    "reporting_requirement": "report_per_seed_only"
                    if degenerate
                    else "pooled_interval_admissible",
                }
            )
    return rows


# ------------------------------------------------------------------- A3 support


def a3_survivor_shift(analysis_root: Path, datasets: list[str]) -> list[dict[str, Any]]:
    """Aggregate the transition-11 score and IoU shift over seeds."""
    rows = []
    for dataset in datasets:
        collected: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for run in SEED_RUNS:
            path = analysis_root / dataset / run / "continuous_shift_summary.csv"
            for row in read_csv(path):
                bucket = collected[(row["scale"], row["metric"])]
                bucket["median"].append(float(row["median"]))
                bucket["q1"].append(float(row["q1"]))
                bucket["q3"].append(float(row["q3"]))
                bucket["p05"].append(float(row["p05"]))
                bucket["p95"].append(float(row["p95"]))
                bucket["negative_fraction"].append(float(row["negative_fraction"]))
                bucket["n"].append(float(row["n"]))
        for metric in ("delta_score", "delta_iou"):
            for scale in SCALES:
                bucket = collected[(scale, metric)]
                median_mean, median_std = mean_std(bucket["median"])
                negative_mean, negative_std = mean_std(bucket["negative_fraction"])
                rows.append(
                    {
                        "dataset": dataset,
                        "scale": scale,
                        "metric": metric,
                        "transition": "11",
                        "n_mean": statistics.mean(bucket["n"]),
                        "median_mean": median_mean,
                        "median_std": median_std,
                        "q1_mean": statistics.mean(bucket["q1"]),
                        "q3_mean": statistics.mean(bucket["q3"]),
                        "p05_mean": statistics.mean(bucket["p05"]),
                        "p95_mean": statistics.mean(bucket["p95"]),
                        "negative_fraction_mean": negative_mean,
                        "negative_fraction_std": negative_std,
                        "seed_n": len(bucket["median"]),
                    }
                )
    return rows


# ------------------------------------------------------------------- A4 support


def a4_flip_shift(
    analysis_root: Path,
    output_root: Path,
    datasets: list[str],
    score_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score of the surviving evidence box for every confidence-mode flip.

    Answers whether a lost detection collapsed far below the operating point or
    merely grazed it, which decides whether the mechanism claim is structural.
    """
    grazing_low = 0.5 * score_threshold
    collapse_high = 0.25 * score_threshold
    per_seed_rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    pooled_fp32: dict[tuple[str, str], list[float]] = defaultdict(list)
    for dataset in datasets:
        for run in SEED_RUNS:
            run_dir = analysis_root / dataset / run
            transitions = read_json(run_dir / "paired_transitions.json")
            fp32_score = {
                (str(row["image_id"]), str(row["object_id"])): float(row["fp32_score"])
                for row in transitions
                if row["transition"] == "10"
            }
            del transitions
            modes = read_json(run_dir / "failure_modes.json")
            wanted = {
                int(row["evidence_prediction_id"])
                for row in modes
                if row["mode"] == "confidence"
                and row["evidence_prediction_id"] is not None
            }
            predictions = read_json(output_root / dataset / run / "predictions.json")
            evidence_score = {
                index: float(predictions[index]["score"]) for index in wanted
            }
            del predictions
            by_scale: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row in modes:
                if row["mode"] != "confidence":
                    continue
                index = row["evidence_prediction_id"]
                key = (str(row["image_id"]), str(row["object_id"]))
                if index is None or key not in fp32_score:
                    continue
                by_scale[row["scale"]].append(
                    (fp32_score[key], evidence_score[int(index)])
                )
            for scale in SCALES:
                pairs = by_scale.get(scale, [])
                before = np.asarray([value[0] for value in pairs], dtype=float)
                after = np.asarray([value[1] for value in pairs], dtype=float)
                delta = after - before
                pooled[(dataset, scale)].extend(after.tolist())
                pooled_fp32[(dataset, scale)].extend(before.tolist())
                per_seed_rows.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "scale": scale,
                        "n_confidence_flips": int(delta.size),
                        "fp32_score_median": float(np.median(before))
                        if before.size
                        else None,
                        "int8_evidence_score_median": float(np.median(after))
                        if after.size
                        else None,
                        "delta_score_median": float(np.median(delta))
                        if delta.size
                        else None,
                        "grazed_fraction": float(
                            np.count_nonzero(after >= grazing_low) / after.size
                        )
                        if after.size
                        else None,
                        "collapsed_fraction": float(
                            np.count_nonzero(after < collapse_high) / after.size
                        )
                        if after.size
                        else None,
                    }
                )
    summary_rows = []
    for dataset in datasets:
        for scale in SCALES:
            after = np.asarray(pooled[(dataset, scale)], dtype=float)
            before = np.asarray(pooled_fp32[(dataset, scale)], dtype=float)
            delta = after - before
            seed_medians = [
                row["delta_score_median"]
                for row in per_seed_rows
                if row["dataset"] == dataset
                and row["scale"] == scale
                and row["delta_score_median"] is not None
            ]
            _, median_std = mean_std(seed_medians)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "score_threshold": score_threshold,
                    "n_confidence_flips_all_seeds": int(after.size),
                    "n_per_seed_mean": after.size / len(SEED_RUNS),
                    "fp32_score_median": float(np.median(before))
                    if before.size
                    else None,
                    "int8_evidence_score_median": float(np.median(after))
                    if after.size
                    else None,
                    "int8_evidence_score_q1": float(np.percentile(after, 25))
                    if after.size
                    else None,
                    "int8_evidence_score_q3": float(np.percentile(after, 75))
                    if after.size
                    else None,
                    "delta_score_median": float(np.median(delta))
                    if delta.size
                    else None,
                    "delta_score_median_seed_std": median_std,
                    "grazed_fraction": float(
                        np.count_nonzero(after >= grazing_low) / after.size
                    )
                    if after.size
                    else None,
                    "collapsed_fraction": float(
                        np.count_nonzero(after < collapse_high) / after.size
                    )
                    if after.size
                    else None,
                    "grazing_band_low": grazing_low,
                    "collapse_band_high": collapse_high,
                }
            )
    return summary_rows, per_seed_rows


def a4_threshold_sweep(
    analysis_root: Path,
    datasets: list[str],
    score_thresholds: list[float],
    iou_thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every frozen score x IoU cell, not just the primary operating point."""
    rows = []
    for dataset in datasets:
        sweeps = {
            run: read_csv(analysis_root / dataset / run / "threshold_sweep.csv")
            for run in SEED_RUNS
        }
        for iou in iou_thresholds:
            for score in score_thresholds:
                by_scale: dict[str, list[float]] = defaultdict(list)
                for values in sweeps.values():
                    for row in values:
                        if (
                            abs(float(row["score_threshold"]) - score) < 1e-12
                            and abs(float(row["iou_threshold"]) - iou) < 1e-12
                        ):
                            qfr = optional_float(row["qfr"])
                            if qfr is not None:
                                by_scale[row["scale"]].append(qfr)
                means = {scale: statistics.mean(by_scale[scale]) for scale in SCALES}
                stds = {scale: statistics.pstdev(by_scale[scale]) for scale in SCALES}
                ordering = sorted(SCALES, key=lambda scale: means[scale])
                rows.append(
                    {
                        "dataset": dataset,
                        "iou_threshold": iou,
                        "score_threshold": score,
                        "qfr_small": means["small"],
                        "qfr_medium": means["medium"],
                        "qfr_large": means["large"],
                        "qfr_small_std": stds["small"],
                        "qfr_medium_std": stds["medium"],
                        "qfr_large_std": stds["large"],
                        "ordering_ascending": ">".join(
                            SHORT[s] for s in ordering[::-1]
                        ),
                        "small_lowest": means["small"]
                        < min(means["medium"], means["large"]),
                        "small_highest": means["small"]
                        > max(means["medium"], means["large"]),
                        "within_3pp_of_both": abs(means["small"] - means["medium"])
                        < 0.03
                        and abs(means["small"] - means["large"]) < 0.03,
                        "seed_n": len(by_scale["small"]),
                    }
                )
    stability = []
    for dataset in datasets:
        for iou in iou_thresholds:
            cells = [
                row
                for row in rows
                if row["dataset"] == dataset and abs(row["iou_threshold"] - iou) < 1e-12
            ]
            orderings = {row["ordering_ascending"] for row in cells}
            stability.append(
                {
                    "dataset": dataset,
                    "iou_threshold": iou,
                    "cells": len(cells),
                    "small_highest_cells": sum(
                        bool(row["small_highest"]) for row in cells
                    ),
                    "small_lowest_cells": sum(
                        bool(row["small_lowest"]) for row in cells
                    ),
                    "distinct_orderings": len(orderings),
                    "orderings": ";".join(sorted(orderings)),
                    "ordering_stable_across_score_thresholds": len(orderings) == 1,
                }
            )
    return rows, stability


# ---------------------------------------------------------------- B1/B2/B3 rows


def b1_pipeline_validation(
    analysis_root: Path, audit_root: Path | None, tolerance: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    control_rows = []
    for row in read_csv(analysis_root / "pipeline_controls.csv"):
        relative = float(row["relative_drop"])
        control_rows.append(
            {
                "dataset": row["dataset"],
                "metric": row["metric"],
                "pytorch_fp32": float(row["pytorch_fp32"]),
                "tensorrt_fp16": float(row["tensorrt_fp16"]),
                "absolute_drop": float(row["absolute_drop"]),
                "relative_drop": relative,
                "abs_relative_drop": abs(relative),
                "within_tolerance": abs(relative) < tolerance,
                "tolerance": tolerance,
            }
        )
    audit_rows: list[dict[str, Any]] = []
    if audit_root is not None and (audit_root / "engine_audit.csv").is_file():
        for row in read_csv(audit_root / "engine_audit.csv"):
            audit_rows.append(
                {
                    "dataset": row["dataset"],
                    "run": row["run"],
                    "declared_precision": row["declared_precision"],
                    "inspection_status": row["inspection_status"],
                    "layer_count": int(row["layer_count"]),
                    "quantize_named_layers": int(row["quantize_named_layers"]),
                    "weight_quantize_named_layers": int(
                        row["weight_quantize_named_layers"]
                    ),
                    "qdq_detected": row["qdq_detected"],
                    "input_data_type": row["input_data_type"],
                    "output_data_type": row["output_data_type"],
                    "per_channel_scheme_resolved": False,
                    "audit_interpretation": row["audit_interpretation"],
                }
            )
    return control_rows, audit_rows


def b2_instance_counts(analysis_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(analysis_root / "validation_statistics.csv"):
        total = int(row["number_of_objects"])
        counts = {scale: int(row[f"{scale}_count"]) for scale in SCALES}
        rows.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "number_of_images": int(row["number_of_images"]),
                "number_of_objects": total,
                "small_count": counts["small"],
                "medium_count": counts["medium"],
                "large_count": counts["large"],
                "small_percent": 100 * counts["small"] / total,
                "medium_percent": 100 * counts["medium"] / total,
                "large_percent": 100 * counts["large"] / total,
                "objects_per_image": total / int(row["number_of_images"]),
                "median_bbox_area": float(row["median_bbox_area"]),
                "median_relative_bbox_area": float(row["median_relative_bbox_area"]),
            }
        )
    return rows


def b3_false_positives(
    analysis_root: Path, datasets: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        collected: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        available = 0
        for run in SEED_RUNS:
            path = analysis_root / dataset / run / "false_positive_counts.csv"
            if not path.is_file():
                continue
            available += 1
            for row in read_csv(path):
                bucket = collected[row["scale"]]
                bucket["fp32_fp"].append(float(row["fp32_fp"]))
                bucket["int8_fp"].append(float(row["int8_fp"]))
                bucket["delta_fp"].append(float(row["delta_fp"]))
                bucket["int8_only_fp"].append(float(row["int8_only_fp"]))
                bucket["fp32_fp_per_image"].append(float(row["fp32_fp_per_image"]))
                bucket["int8_fp_per_image"].append(float(row["int8_fp_per_image"]))
                fraction = optional_float(row["int8_only_fraction"])
                if fraction is not None:
                    bucket["int8_only_fraction"].append(fraction)
        if not available:
            continue
        for scale in SCALES:
            bucket = collected[scale]
            int8_mean, int8_std = mean_std(bucket["int8_fp"])
            rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "fp32_fp": statistics.mean(bucket["fp32_fp"]),
                    "int8_fp_mean": int8_mean,
                    "int8_fp_std": int8_std,
                    "delta_fp_mean": statistics.mean(bucket["delta_fp"]),
                    "int8_only_fp_mean": statistics.mean(bucket["int8_only_fp"]),
                    "int8_only_fraction_mean": statistics.mean(
                        bucket["int8_only_fraction"]
                    )
                    if bucket["int8_only_fraction"]
                    else None,
                    "fp32_fp_per_image": statistics.mean(bucket["fp32_fp_per_image"]),
                    "int8_fp_per_image_mean": statistics.mean(
                        bucket["int8_fp_per_image"]
                    ),
                    "seed_n": available,
                }
            )
    return rows


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--resolution-id", required=True)
    parser.add_argument(
        "--compare-analysis",
        help="Earlier analysis directory used for the A2 baseline comparison",
    )
    parser.add_argument("--engine-audit-root")
    parser.add_argument(
        "--fp16-relative-tolerance",
        type=float,
        default=0.01,
        help="Pass band for the FP16 control, as a relative AP drop",
    )
    parser.add_argument(
        "--required-common-bins",
        type=int,
        default=8,
        help="Frozen coverage requirement applied to the decile stratification",
    )
    parser.add_argument(
        "--minimum-flip-events",
        type=int,
        default=50,
        help="Flip events below which a scale coefficient is reported per seed only",
    )
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=0.25,
        help=(
            "Share of a scale that must sit at or below the reference scale's p95 "
            "confidence for its coefficient to be read as a seed mean"
        ),
    )
    args = parser.parse_args()
    if (
        not args.resolution_id.strip()
        or "/" in args.resolution_id
        or ".." in args.resolution_id
    ):
        raise ValueError("--resolution-id must be a non-empty directory-safe name")
    analysis_root = Path(args.analysis_root).resolve()
    if not analysis_root.is_dir():
        raise FileNotFoundError(f"Analysis root does not exist: {analysis_root}")
    manifest_path = analysis_root / "analysis_manifest.json"
    analysis_manifest = read_json(manifest_path)
    if analysis_manifest["failures"]:
        raise ValueError("Refusing to resolve issues from an analysis with failures")
    protocol_path = analysis_root / "protocol_resolved.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["protocol"]
    config = yaml.safe_load(
        (analysis_root / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    output_root = Path(config["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    datasets = sorted(
        path.name
        for path in analysis_root.iterdir()
        if path.is_dir() and (path / "validation_statistics.json").is_file()
    )
    if not datasets:
        raise ValueError("No completed dataset analysis directories found")

    root = output_root / "issue_resolution" / args.resolution_id
    root.mkdir(parents=True, exist_ok=False)

    minimum = int(protocol["stratification"]["minimum_per_scale_for_decision"])
    primary_score = float(protocol["primary_score"])
    score_thresholds = [float(value) for value in protocol["score_thresholds"]]
    iou_thresholds = [float(value) for value in protocol["iou_thresholds"]]

    # ---------------------------------------------------------------- A1
    histogram_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    concentration: dict[str, Any] = {}
    for dataset in datasets:
        transitions = read_json(
            analysis_root / dataset / SEED_RUNS[0] / "paired_transitions.json"
        )
        per_scale = fp32_success_scores(transitions)
        del transitions
        frozen_edges = read_json(analysis_root / dataset / "confidence_bin_edges.json")
        recomputed = pooled_edges(per_scale, int(protocol["stratification"]["bins"]))
        histogram_rows.extend(a1_histogram(dataset, per_scale, frozen_edges, minimum))
        distribution_rows.extend(a1_distribution(dataset, per_scale))
        variant_rows.extend(
            a1_binning_variants(
                dataset,
                per_scale,
                minimum,
                int(protocol["stratification"]["bins"]),
                args.required_common_bins,
            )
        )
        support_rows.extend(a1_common_support(dataset, per_scale))
        concentration[dataset] = {
            "top_two_decile_share": a1_top_bin_concentration(per_scale, frozen_edges),
            "frozen_edges_reproduced": bool(
                np.allclose(frozen_edges, recomputed, atol=1e-9)
            ),
        }
    write_csv(root / "a1_score_histogram_by_scale.csv", histogram_rows)
    write_csv(root / "a1_score_distribution_by_scale.csv", distribution_rows)
    write_csv(root / "a1_binning_variants.csv", variant_rows)
    write_csv(root / "a1_common_support.csv", support_rows)

    a1_diagnosis = {
        "question": "Why does the frozen decile stratification lose coverage?",
        "mechanism_tested": (
            "Decile edges are cut on the pooled FP32-matched score distribution "
            "while each scale occupies a different part of that distribution."
        ),
        "evidence": concentration,
        "frozen_required_common_bins": args.required_common_bins,
        "usable_bins_by_variant": {row["dataset"]: {} for row in variant_rows},
        "note": (
            "Coarser binning is reported as a post-hoc variant carrying the same "
            f"{args.required_common_bins}-of-"
            f"{int(protocol['stratification']['bins'])} coverage ratio. The "
            "frozen decile protocol and its requirement were not changed."
        ),
    }
    for row in variant_rows:
        a1_diagnosis["usable_bins_by_variant"][row["dataset"]][str(row["bins"])] = {
            "valid_common_bins": row["valid_common_bins"],
            "required_for_decision": row["required_for_decision"],
            "coverage_available": row["coverage_available"],
            "status": row["status"],
        }
    write_json(root / "a1_diagnosis.json", a1_diagnosis)

    # ---------------------------------------------------------------- A2
    comparison_root = (
        Path(args.compare_analysis).resolve() if args.compare_analysis else None
    )
    a2 = a2_baseline_evidence(output_root, analysis_root, comparison_root, datasets)
    a2["verdict"] = {
        "inference_deterministic_all_datasets": all(
            record["inference_bit_identical"] for record in a2["per_dataset"].values()
        ),
        "total_drop_reconciled_all_datasets": all(
            row["agrees_within_1e_9"] for row in a2["total_drop_reconciliation"]
        ),
        "single_baseline": "analysis pycocotools COCOeval on fp32/predictions.json",
    }
    write_json(root / "a2_fp32_baseline_provenance.json", a2)
    if comparison_root is not None:
        write_csv(
            root / "a2_cross_analysis_comparison.csv",
            a2["cross_analysis_comparison"]["rows"],
        )
    write_csv(
        root / "a2_total_drop_reconciliation.csv", a2["total_drop_reconciliation"]
    )

    # ---------------------------------------------------------------- A3 / A4
    write_csv(
        root / "a3_survivor_score_shift.csv", a3_survivor_shift(analysis_root, datasets)
    )
    flip_summary, flip_per_seed = a4_flip_shift(
        analysis_root, output_root, datasets, primary_score
    )
    write_csv(root / "a4_flip_score_shift.csv", flip_summary)
    write_csv(root / "a4_flip_score_shift_per_seed.csv", flip_per_seed)
    sweep_rows, sweep_stability = a4_threshold_sweep(
        analysis_root, datasets, score_thresholds, iou_thresholds
    )
    write_csv(root / "a4_threshold_sweep_ordering.csv", sweep_rows)
    write_csv(root / "a4_threshold_sweep_stability.csv", sweep_stability)

    # ---------------------------------------------------------------- B1 / B2 / B3
    audit_root = (
        Path(args.engine_audit_root).resolve() if args.engine_audit_root else None
    )
    control_rows, audit_rows = b1_pipeline_validation(
        analysis_root, audit_root, args.fp16_relative_tolerance
    )
    write_csv(root / "b1_fp16_control.csv", control_rows)
    if audit_rows:
        write_csv(root / "b1_engine_audit_summary.csv", audit_rows)
    write_csv(root / "b2_gt_instance_counts.csv", b2_instance_counts(analysis_root))
    fp_rows = b3_false_positives(analysis_root, datasets)
    write_csv(root / "b3_false_positive_accounting.csv", fp_rows)

    # ------------------------------------------------------------- C1 / C2 / C3
    per_seed_regression, regression_verdicts = regression_interpretability(
        analysis_root, datasets, args.minimum_flip_events, args.minimum_overlap
    )
    write_csv(root / "c1_regression_per_seed.csv", per_seed_regression)
    write_csv(root / "c1_regression_interpretability.csv", regression_verdicts)
    variance_rows = variance_decomposition(analysis_root, datasets)
    write_csv(root / "c2_variance_decomposition.csv", variance_rows)
    write_csv(
        root / "c3_qfr_bootstrap_ci.csv",
        qfr_bootstrap_intervals(analysis_root, datasets),
    )

    # ---------------------------------------------------------------- summary
    status = {
        "A1": "resolved_with_diagnosis"
        if all(entry["frozen_edges_reproduced"] for entry in concentration.values())
        else "frozen_edges_did_not_reproduce",
        "A2": "resolved"
        if a2["verdict"]["inference_deterministic_all_datasets"]
        and a2["verdict"]["total_drop_reconciled_all_datasets"]
        else "unresolved",
        "A3": "resolved",
        "A4": "resolved",
        "B1": "pass"
        if all(row["within_tolerance"] for row in control_rows)
        else "fail",
        "B2": "resolved",
        "B3": "resolved" if len(fp_rows) == len(datasets) * len(SCALES) else "partial",
        "B4": "justified_by_c2_requires_gpu_not_run_here"
        if any(row["calibration_ablation_could_help"] for row in variance_rows)
        else "not_justified_by_c2",
        "C1": "terms_withheld_from_seed_mean"
        if any(not row["interpretable_as_seed_mean"] for row in regression_verdicts)
        else "all_terms_interpretable",
        "C2": "resolved",
        "C3": "qfr_intervals_resolved_ap_intervals_pending_bootstrap_ap_ci",
    }
    write_json(
        root / "resolution_summary.json",
        {
            "resolution_id": args.resolution_id,
            "analysis_root": str(analysis_root),
            "analysis_manifest_sha256": sha256(manifest_path),
            "protocol_sha256": sha256(protocol_path),
            "comparison_analysis": str(comparison_root) if comparison_root else None,
            "engine_audit_root": str(audit_root) if audit_root else None,
            "datasets": datasets,
            "seed_runs": list(SEED_RUNS),
            "primary_score": primary_score,
            "status": status,
            "notes": [
                "No inference, engine build, or evaluation of new predictions was run.",
                (
                    "The frozen protocol was not modified; coarser stratification "
                    "binning is reported only as a labelled post-hoc variant."
                ),
                (
                    "B4 calibration-size ablation needs GPU inference and is out "
                    "of scope for this offline pass; c2 records whether it could "
                    "reduce the observed variance at all."
                ),
                (
                    "Scale coefficients whose per-seed intervals disagree in sign "
                    "are reported per seed and never as a seed mean."
                ),
                (
                    "c3 covers QFR only. Scale-wise AP intervals come from "
                    "scripts/bootstrap_ap_ci.py, which re-accumulates official "
                    "COCOeval over resampled images."
                ),
            ],
        },
    )
    outputs = sorted(path for path in root.rglob("*") if path.is_file())
    write_json(
        root / "resolution_manifest.json",
        {
            "resolution_id": args.resolution_id,
            "outputs": [
                {"path": str(path), "sha256": sha256(path)} for path in outputs
            ],
        },
    )
    print(json.dumps({"resolution_root": str(root), "status": status}, indent=2))


if __name__ == "__main__":
    main()
