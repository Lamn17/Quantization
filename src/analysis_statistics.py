"""Image-clustered uncertainty and adjusted flip-rate models for frozen analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from diagnostics import SCALES, summarize_transitions


def _design(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in rows if row["fp32_detected"]]
    if not selected:
        raise ValueError("Regression requires FP32-success instances")
    scores = np.asarray([float(row["fp32_score"]) for row in selected])
    ious = np.asarray([float(row["fp32_iou"]) for row in selected])

    def standardized(values: np.ndarray) -> np.ndarray:
        std = float(values.std())
        return (values - values.mean()) / std if std > 0 else values * 0.0

    x = np.column_stack(
        [
            np.ones(len(selected)),
            [row["scale"] == "medium" for row in selected],
            [row["scale"] == "large" for row in selected],
            standardized(scores),
            standardized(ious),
        ]
    ).astype(float)
    y = np.asarray([row["transition"] == "10" for row in selected], dtype=float)
    return (
        x,
        y,
        ["intercept", "scale_medium", "scale_large", "fp32_score_z", "fp32_iou_z"],
    )


def logistic_adjustment(
    rows: list[dict[str, Any]], maximum_iterations: int = 100, tolerance: float = 1e-8
) -> dict[str, Any]:
    """Fit a deterministic logistic model with small ridge stabilization."""
    x, y, names = _design(rows)
    beta = np.zeros(x.shape[1])
    ridge = np.diag([0.0] + [1e-8] * (x.shape[1] - 1))
    converged = False
    for iteration in range(1, maximum_iterations + 1):
        eta = np.clip(x @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        information = x.T @ (weights[:, None] * x) + ridge
        score = x.T @ (y - probability) - ridge @ beta
        step = np.linalg.pinv(information) @ score
        beta += step
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "n": len(y),
        "events": int(y.sum()),
        "converged": converged,
        "iterations": iteration,
        "reference_scale": "small",
        "coefficients": {
            name: {
                "log_odds": float(value),
                "odds_ratio": float(np.exp(np.clip(value, -30, 30))),
                "standard_error_model_based": float(se),
            }
            for name, value, se in zip(names, beta, standard_errors)
        },
    }


def image_cluster_bootstrap(
    rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Bootstrap QFR and adjusted scale coefficients by resampling images."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(row)
    image_ids = sorted(grouped)
    if not image_ids or replicates <= 0:
        raise ValueError("Bootstrap requires images and a positive replicate count")
    rng = np.random.default_rng(seed)
    qfr_values = {scale: [] for scale in SCALES}
    coefficient_values = {name: [] for name in ("scale_medium", "scale_large")}
    failed_regressions = 0
    for _ in range(replicates):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        replicate_rows = [row for image_id in sampled for row in grouped[str(image_id)]]
        summary = summarize_transitions(replicate_rows)
        for scale in SCALES:
            value = summary[scale]["qfr"]
            if value is not None:
                qfr_values[scale].append(value)
        try:
            model = logistic_adjustment(replicate_rows, maximum_iterations, tolerance)
            if not model["converged"]:
                failed_regressions += 1
                continue
            for name, values in coefficient_values.items():
                values.append(model["coefficients"][name]["log_odds"])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failed_regressions += 1

    def interval(values: list[float]) -> dict[str, float | int | None]:
        return {
            "n": len(values),
            "lower_95": float(np.percentile(values, 2.5)) if values else None,
            "median": float(np.percentile(values, 50)) if values else None,
            "upper_95": float(np.percentile(values, 97.5)) if values else None,
        }

    return {
        "unit": "image",
        "requested_replicates": replicates,
        "seed": seed,
        "failed_regressions": failed_regressions,
        "qfr": {scale: interval(values) for scale, values in qfr_values.items()},
        "regression_log_odds": {
            name: interval(values) for name, values in coefficient_values.items()
        },
    }
