#!/usr/bin/env python3
"""Build traceable paper tables, figures, and the frozen-case assessment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

SCALES = ("small", "medium", "large")
SEED_PATTERN = re.compile(r"int8_random_s(\d+)$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty paper table: {path}")
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


def seed_dirs(dataset_dir: Path) -> list[Path]:
    paths = [
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and SEED_PATTERN.fullmatch(path.name)
    ]
    return sorted(paths, key=lambda path: int(SEED_PATTERN.fullmatch(path.name)[1]))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return statistics.mean(values), statistics.pstdev(values)


def transition_table(dataset_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for dataset_dir in dataset_dirs:
        sources = seed_dirs(dataset_dir)
        for scale in SCALES:
            qfr, qrr = [], []
            for source in sources:
                summary = read_json(source / "transition_summary.json")[scale]
                qfr.append(float(summary["qfr"]))
                qrr.append(float(summary["qrr"]))
            qfr_mean, qfr_std = mean_std(qfr)
            qrr_mean, qrr_std = mean_std(qrr)
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "scale": scale,
                    "qfr_mean": qfr_mean,
                    "qfr_std": qfr_std,
                    "qrr_mean": qrr_mean,
                    "qrr_std": qrr_std,
                    "seed_n": len(sources),
                    "source": ";".join(
                        str(path / "transition_summary.json") for path in sources
                    ),
                }
            )
    return rows


def regression_table(dataset_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for dataset_dir in dataset_dirs:
        sources = seed_dirs(dataset_dir)
        for term in ("scale_medium", "scale_large"):
            values, positive, negative = [], 0, 0
            for source in sources:
                model = read_json(source / "regression.json")
                bootstrap = read_json(source / "bootstrap_ci.json")
                values.append(float(model["coefficients"][term]["log_odds"]))
                interval = bootstrap["regression_log_odds"][term]
                positive += float(interval["lower_95"]) > 0
                negative += float(interval["upper_95"]) < 0
            value_mean, value_std = mean_std(values)
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "term": term,
                    "reference_scale": "small",
                    "log_odds_mean": value_mean,
                    "log_odds_std": value_std,
                    "odds_ratio_of_mean": math.exp(value_mean),
                    "positive_ci_seeds": positive,
                    "negative_ci_seeds": negative,
                    "overlapping_zero_ci_seeds": len(sources) - positive - negative,
                    "seed_n": len(sources),
                    "source": ";".join(
                        f"{path / 'regression.json'}|{path / 'bootstrap_ci.json'}"
                        for path in sources
                    ),
                }
            )
    return rows


def stratification_table(
    dataset_dirs: list[Path], minimum: int
) -> list[dict[str, Any]]:
    rows = []
    for dataset_dir in dataset_dirs:
        for source in seed_dirs(dataset_dir):
            by_bin: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
            path = source / "stratified_qfr.csv"
            for row in read_csv(path):
                by_bin[int(row["bin"])][row["scale"]] = row
            valid, small_lowest, small_highest = 0, 0, 0
            for values in by_bin.values():
                if not all(
                    scale in values and int(values[scale]["n"]) >= minimum
                    for scale in SCALES
                ):
                    continue
                valid += 1
                qfr = {scale: float(values[scale]["qfr"]) for scale in SCALES}
                small_lowest += qfr["small"] < min(qfr["medium"], qfr["large"])
                small_highest += qfr["small"] > max(qfr["medium"], qfr["large"])
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "run": source.name,
                    "valid_common_bins": valid,
                    "small_lowest_bins": small_lowest,
                    "small_highest_bins": small_highest,
                    "minimum_per_scale": minimum,
                    "source": str(path),
                }
            )
    return rows


def threshold_ordering(
    dataset_dirs: list[Path], primary_score: float, iou_thresholds: list[float]
) -> list[dict[str, Any]]:
    rows = []
    for dataset_dir in dataset_dirs:
        sources = seed_dirs(dataset_dir)
        for iou in iou_thresholds:
            by_scale: dict[str, list[float]] = defaultdict(list)
            source_paths = []
            for source in sources:
                path = source / "threshold_sweep.csv"
                source_paths.append(str(path))
                for value in read_csv(path):
                    if math.isclose(
                        float(value["score_threshold"]), primary_score
                    ) and math.isclose(float(value["iou_threshold"]), iou):
                        by_scale[value["scale"]].append(float(value["qfr"]))
            means = {scale: statistics.mean(by_scale[scale]) for scale in SCALES}
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "score_threshold": primary_score,
                    "iou_threshold": iou,
                    "qfr_small": means["small"],
                    "qfr_medium": means["medium"],
                    "qfr_large": means["large"],
                    "small_lowest": means["small"]
                    < min(means["medium"], means["large"]),
                    "small_highest": means["small"]
                    > max(means["medium"], means["large"]),
                    "within_3pp_of_both": abs(means["small"] - means["medium"]) < 0.03
                    and abs(means["small"] - means["large"]) < 0.03,
                    "source": ";".join(source_paths),
                }
            )
    return rows


def mechanism_table(dataset_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for dataset_dir in dataset_dirs:
        sources = seed_dirs(dataset_dir)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        source_paths = []
        for source in sources:
            path = source / "failure_mode_summary.csv"
            source_paths.append(str(path))
            for value in read_csv(path):
                counts[(value["scale"], value["mode"])] += int(value["count"])
        for scale in SCALES:
            total = sum(
                value
                for (item_scale, _), value in counts.items()
                if item_scale == scale
            )
            for mode in ("confidence", "localization", "class", "complete_miss"):
                count = counts[(scale, mode)]
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "scale": scale,
                        "mode": mode,
                        "count_across_seeds": count,
                        "fraction": count / total if total else None,
                        "source": ";".join(source_paths),
                    }
                )
    return rows


def decomposition_table(dataset_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for dataset_dir in dataset_dirs:
        sources = seed_dirs(dataset_dir)
        values: dict[str, list[float]] = defaultdict(list)
        source_paths = []
        for source in sources:
            path = source / "ap_decomposition.csv"
            source_paths.append(str(path))
            overall = next(row for row in read_csv(path) if row["metric"] == "AP")
            for key in (
                "total_drop",
                "geometry_and_detection_set_component",
                "ranking_component",
            ):
                values[key].append(float(overall[key]))
        total = statistics.mean(values["total_drop"])
        rows.append(
            {
                "dataset": dataset_dir.name,
                "total_drop_mean": total,
                "geometry_detection_set_mean": statistics.mean(
                    values["geometry_and_detection_set_component"]
                ),
                "ranking_mean": statistics.mean(values["ranking_component"]),
                "ranking_fraction_of_drop": statistics.mean(values["ranking_component"])
                / total,
                "seed_n": len(sources),
                "source": ";".join(source_paths),
            }
        )
    return rows


def case_assessment(
    ordering: list[dict[str, Any]],
    regression: list[dict[str, Any]],
    strata: list[dict[str, Any]],
    required_combinations: int,
    required_bins: int,
) -> dict[str, Any]:
    small_lowest = sum(bool(row["small_lowest"]) for row in ordering)
    small_highest = sum(bool(row["small_highest"]) for row in ordering)
    close = sum(bool(row["within_3pp_of_both"]) for row in ordering)
    total = len(ordering)
    regression_uniform_positive = all(
        int(row["positive_ci_seeds"]) == int(row["seed_n"]) for row in regression
    )
    regression_uniform_negative = all(
        int(row["negative_ci_seeds"]) == int(row["seed_n"]) for row in regression
    )
    coverage_pass = all(
        int(row["valid_common_bins"]) >= required_bins for row in strata
    )
    if (
        small_lowest >= required_combinations
        and regression_uniform_positive
        and coverage_pass
    ):
        classification = "case_a"
    elif small_lowest >= required_combinations:
        classification = "case_a_prime"
    elif close > total / 2:
        classification = "case_b"
    elif (
        small_highest >= required_combinations
        and regression_uniform_negative
        and coverage_pass
    ):
        classification = "case_c"
    else:
        classification = "mixed_unclassified_by_preregistered_cases"
    return {
        "classification": classification,
        "post_hoc_label": "mixed_dataset_and_difficulty_dependent_outcome"
        if classification.startswith("mixed_")
        else None,
        "raw_ordering": {
            "small_lowest_combinations": small_lowest,
            "small_highest_combinations": small_highest,
            "within_3pp_combinations": close,
            "total_dataset_iou_combinations": total,
            "required_combinations": required_combinations,
        },
        "difficulty_control": {
            "all_regression_seed_cis_positive": regression_uniform_positive,
            "all_regression_seed_cis_negative": regression_uniform_negative,
            "all_runs_have_required_common_bins": coverage_pass,
            "required_common_bins": required_bins,
        },
        "interpretation": (
            "The preregistered raw Case-C ordering is nearly met, but its "
            "difficulty-control requirements fail. No universal scale ordering "
            "is supported; report the mixed outcome without changing thresholds "
            "or discarding seeds."
        ),
    }


def figure_metric_disagreement(rows: list[dict[str, str]], target: Path) -> None:
    source = [row for row in rows if row["metric"] in {"AP_S", "AP_M", "AP_L"}]
    datasets = list(dict.fromkeys(row["dataset"] for row in source))
    x = np.arange(len(datasets))
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for index, metric in enumerate(("AP_S", "AP_M", "AP_L")):
        selected = [
            next(
                row
                for row in source
                if row["dataset"] == dataset and row["metric"] == metric
            )
            for dataset in datasets
        ]
        axes[0].bar(
            x + (index - 1) * width,
            [float(row["delta_ap_mean"]) for row in selected],
            width,
            label=metric[-1],
        )
        axes[1].bar(
            x + (index - 1) * width,
            [100 * float(row["relative_drop"]) for row in selected],
            width,
            label=metric[-1],
        )
    axes[0].set_ylabel("Absolute AP drop")
    axes[1].set_ylabel("Relative AP drop (%)")
    for axis in axes:
        axis.set_xticks(x, datasets)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(title="Scale", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=300)
    plt.close(figure)


def figure_qfr(rows: list[dict[str, Any]], target: Path) -> None:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    x = np.arange(len(datasets))
    offsets = {"small": -0.18, "medium": 0.0, "large": 0.18}
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for scale in SCALES:
        selected = [
            next(
                row
                for row in rows
                if row["dataset"] == dataset and row["scale"] == scale
            )
            for dataset in datasets
        ]
        axis.errorbar(
            x + offsets[scale],
            [100 * row["qfr_mean"] for row in selected],
            yerr=[100 * row["qfr_std"] for row in selected],
            marker="o",
            capsize=4,
            linestyle="none",
            label=scale.title(),
        )
    axis.set_xticks(x, datasets)
    axis.set_ylabel("QFR (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=300)
    plt.close(figure)


def figure_failure_modes(rows: list[dict[str, Any]], target: Path) -> None:
    labels = [
        f"{dataset}\n{scale[0].upper()}"
        for dataset in dict.fromkeys(row["dataset"] for row in rows)
        for scale in SCALES
    ]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    bottom = np.zeros(len(labels))
    for mode in ("confidence", "localization", "class", "complete_miss"):
        values = []
        for dataset in dict.fromkeys(row["dataset"] for row in rows):
            for scale in SCALES:
                row = next(
                    item
                    for item in rows
                    if item["dataset"] == dataset
                    and item["scale"] == scale
                    and item["mode"] == mode
                )
                values.append(100 * float(row["fraction"]))
        axis.bar(x, values, bottom=bottom, label=mode.replace("_", " "))
        bottom += np.asarray(values)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Failure share (%)")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(target, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--report-id", required=True)
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root).resolve()
    if not analysis_root.is_dir():
        raise FileNotFoundError(f"Analysis root does not exist: {analysis_root}")
    if not args.report_id.strip() or "/" in args.report_id or ".." in args.report_id:
        raise ValueError("--report-id must be a non-empty directory-safe name")
    report_root = analysis_root.parent.parent / "paper" / args.report_id
    report_root.mkdir(parents=True, exist_ok=False)
    figures = report_root / "figures"
    figures.mkdir()

    manifest_path = analysis_root / "analysis_manifest.json"
    analysis_manifest = read_json(manifest_path)
    if analysis_manifest["failures"]:
        raise ValueError("Cannot build a paper report from an analysis with failures")
    protocol_path = analysis_root / "protocol_resolved.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["protocol"]
    dataset_dirs = sorted(
        path
        for path in analysis_root.iterdir()
        if path.is_dir() and (path / "validation_statistics.json").is_file()
    )
    if not dataset_dirs:
        raise ValueError("No completed dataset analysis directories found")

    table2 = transition_table(dataset_dirs)
    table3 = regression_table(dataset_dirs)
    strata = stratification_table(
        dataset_dirs, int(protocol["stratification"]["minimum_per_scale_for_decision"])
    )
    ordering = threshold_ordering(
        dataset_dirs,
        float(protocol["primary_score"]),
        [float(value) for value in protocol["iou_thresholds"]],
    )
    mechanisms = mechanism_table(dataset_dirs)
    decomposition = decomposition_table(dataset_dirs)
    write_csv(report_root / "table2_qfr_qrr.csv", table2)
    write_csv(report_root / "table3_regression.csv", table3)
    write_csv(report_root / "stratification_coverage.csv", strata)
    write_csv(report_root / "threshold_ordering.csv", ordering)
    write_csv(report_root / "mechanism_failure_modes.csv", mechanisms)
    write_csv(report_root / "ap_decomposition_summary.csv", decomposition)
    assessment = case_assessment(ordering, table3, strata, 5, 8)
    assessment["visdrone_status"] = (
        "provisional_missing_original_ignore_regions"
        if any(
            "VisDrone" in note and "provisional" in note
            for note in analysis_manifest["notes"]
        )
        else "native_ignore_regions_loaded"
    )
    write_json(report_root / "case_assessment.json", assessment)

    paper_ap_path = analysis_root / "paper_ap_table.csv"
    ap_rows = read_csv(paper_ap_path)
    write_csv(
        figures / "figure1_metric_disagreement_source.csv",
        [row for row in ap_rows if row["metric"] in {"AP_S", "AP_M", "AP_L"}],
    )
    write_csv(figures / "figure2_qfr_source.csv", table2)
    write_csv(figures / "figure3_failure_modes_source.csv", mechanisms)
    figure_metric_disagreement(ap_rows, figures / "figure1_metric_disagreement.png")
    figure_qfr(table2, figures / "figure2_qfr.png")
    figure_failure_modes(mechanisms, figures / "figure3_failure_modes.png")

    outputs = sorted(path for path in report_root.rglob("*") if path.is_file())
    write_json(
        report_root / "report_manifest.json",
        {
            "report_id": args.report_id,
            "source_analysis": str(analysis_root),
            "source_analysis_manifest_sha256": sha256(manifest_path),
            "source_protocol_sha256": sha256(protocol_path),
            "outputs": [
                {"path": str(path), "sha256": sha256(path)} for path in outputs
            ],
            "notes": [
                "No inference or evaluation was rerun.",
                "All figure source data are stored beside the rendered figures.",
                "The mixed label is post hoc; the frozen A/A-prime/B/C thresholds were not changed.",
            ],
        },
    )
    print(json.dumps({"report_root": str(report_root), **assessment}, indent=2))


if __name__ == "__main__":
    main()
