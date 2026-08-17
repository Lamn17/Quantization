"""Dataset-independent FP32-versus-INT8 object diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise
from statistics import mean, pstdev
from typing import Any

import numpy as np

from dataset import classify_object_scale
from evaluation import bbox_iou

SCALES = ("small", "medium", "large")


def _prediction_records(
    predictions: list[dict[str, Any]], score_threshold: float
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Index predictions without losing their stable input-list identity."""
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction_id, prediction in enumerate(predictions):
        score = float(prediction["score"])
        if score >= score_threshold:
            index[(str(prediction["image_id"]), int(prediction["class_id"]))].append(
                {**prediction, "prediction_id": prediction_id, "score": score}
            )
    for values in index.values():
        values.sort(key=lambda item: (-item["score"], item["prediction_id"]))
    return index


def greedy_match_model_to_gt(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    score_threshold: float,
    iou_threshold: float,
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[int]]:
    """COCO-like score-ordered, exact-class, one-to-one matching.

    Returns matches keyed by (string image id, string object id), plus the stable
    IDs of unmatched predictions above the score threshold.
    """
    prediction_index = _prediction_records(predictions, score_threshold)
    gt_index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for image in ground_truth:
        for obj in image["objects"]:
            if not obj.get("iscrowd", False):
                gt_index[(str(image["image_id"]), int(obj["class_id"]))].append(obj)
    matches: dict[tuple[str, str], dict[str, Any]] = {}
    matched_prediction_ids: set[int] = set()
    for key, model_predictions in prediction_index.items():
        objects = gt_index.get(key, [])
        available = set(range(len(objects)))
        for prediction in model_predictions:
            candidates = [
                (bbox_iou(prediction["bbox_xyxy"], objects[index]["bbox_xyxy"]), index)
                for index in available
            ]
            best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
            if best is None or best[0] < iou_threshold:
                continue
            iou, object_index = best
            available.remove(object_index)
            obj = objects[object_index]
            matches[(key[0], str(obj["object_id"]))] = {
                "prediction_id": prediction["prediction_id"],
                "score": prediction["score"],
                "iou": iou,
                "bbox_xyxy": [float(value) for value in prediction["bbox_xyxy"]],
            }
            matched_prediction_ids.add(prediction["prediction_id"])
    considered = {
        item["prediction_id"] for values in prediction_index.values() for item in values
    }
    return matches, considered - matched_prediction_ids


def paired_gt_transitions(
    dataset_name: str,
    ground_truth: list[dict[str, Any]],
    fp32_predictions: list[dict[str, Any]],
    int8_predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
    score_threshold: float,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
    """Build the auditable 11/10/01/00 transition table for every scored GT."""
    fp_matches, fp_unmatched = greedy_match_model_to_gt(
        ground_truth, fp32_predictions, score_threshold, iou_threshold
    )
    int8_matches, int8_unmatched = greedy_match_model_to_gt(
        ground_truth, int8_predictions, score_threshold, iou_threshold
    )
    rows: list[dict[str, Any]] = []
    for image in ground_truth:
        for obj in image["objects"]:
            if obj.get("iscrowd", False):
                continue
            key = (str(image["image_id"]), str(obj["object_id"]))
            fp, quant = fp_matches.get(key), int8_matches.get(key)
            fp_detected, int8_detected = fp is not None, quant is not None
            rows.append(
                {
                    "dataset": dataset_name,
                    "image_id": image["image_id"],
                    "object_id": obj["object_id"],
                    "class_id": int(obj["class_id"]),
                    "scale": classify_object_scale(
                        float(obj["area"]), image["width"], image["height"], scale_cfg
                    ),
                    "gt_area": float(obj["area"]),
                    "gt_bbox_xyxy": [float(value) for value in obj["bbox_xyxy"]],
                    "transition": f"{int(fp_detected)}{int(int8_detected)}",
                    "fp32_detected": fp_detected,
                    "int8_detected": int8_detected,
                    "fp32_prediction_id": None if fp is None else fp["prediction_id"],
                    "int8_prediction_id": None
                    if quant is None
                    else quant["prediction_id"],
                    "fp32_score": None if fp is None else fp["score"],
                    "int8_score": None if quant is None else quant["score"],
                    "fp32_iou": None if fp is None else fp["iou"],
                    "int8_iou": None if quant is None else quant["iou"],
                    "fp32_bbox_xyxy": None if fp is None else fp["bbox_xyxy"],
                    "int8_bbox_xyxy": None if quant is None else quant["bbox_xyxy"],
                }
            )
    return rows, {"fp32": fp_unmatched, "int8": int8_unmatched}


def summarize_transitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scale in SCALES:
        selected = [row for row in rows if row["scale"] == scale]
        counts = {
            transition: sum(row["transition"] == transition for row in selected)
            for transition in ("11", "10", "01", "00")
        }
        fp_success = counts["11"] + counts["10"]
        fp_failure = counts["01"] + counts["00"]
        output[scale] = {
            "n_gt": len(selected),
            **counts,
            "qfr": counts["10"] / fp_success if fp_success else None,
            "qrr": counts["01"] / fp_failure if fp_failure else None,
            "net_recall_damage": (counts["10"] - counts["01"]) / len(selected)
            if selected
            else None,
        }
    return output


def confidence_bin_edges(rows: list[dict[str, Any]], bins: int = 10) -> list[float]:
    scores = [float(row["fp32_score"]) for row in rows if row["fp32_detected"]]
    if not scores:
        raise ValueError("Cannot stratify without FP32-success scores")
    return [float(value) for value in np.quantile(scores, np.linspace(0, 1, bins + 1))]


def stratified_qfr(
    rows: list[dict[str, Any]], edges: list[float]
) -> list[dict[str, Any]]:
    if len(edges) < 2 or any(right < left for left, right in pairwise(edges)):
        raise ValueError("Stratification edges must be sorted")
    output = []
    for index, (left, right) in enumerate(pairwise(edges)):
        for scale in SCALES:
            selected = []
            for row in rows:
                if row["scale"] != scale or not row["fp32_detected"]:
                    continue
                score = float(row["fp32_score"])
                if (left <= score < right) or (
                    index == len(edges) - 2 and score == right
                ):
                    selected.append(row)
            lost = sum(row["transition"] == "10" for row in selected)
            output.append(
                {
                    "bin": index + 1,
                    "left": left,
                    "right": right,
                    "scale": scale,
                    "n": len(selected),
                    "lost": lost,
                    "qfr": lost / len(selected) if selected else None,
                }
            )
    return output


def _predicted_scale(
    prediction: dict[str, Any], image: dict[str, Any], scale_cfg: dict[str, Any]
) -> str:
    x1, y1, x2, y2 = map(float, prediction["bbox_xyxy"])
    return classify_object_scale(
        max(0.0, x2 - x1) * max(0.0, y2 - y1),
        int(image["width"]),
        int(image["height"]),
        scale_cfg,
    )


def false_positive_counts(
    ground_truth: list[dict[str, Any]],
    fp32_predictions: list[dict[str, Any]],
    int8_predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
    score_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Count unmatched predictions; ignored-region handling follows loaded GT."""
    _, fp_ids = greedy_match_model_to_gt(
        ground_truth, fp32_predictions, score_threshold, iou_threshold
    )
    _, int8_ids = greedy_match_model_to_gt(
        ground_truth, int8_predictions, score_threshold, iou_threshold
    )
    images = {str(image["image_id"]): image for image in ground_truth}
    counts = {model: {scale: 0 for scale in SCALES} for model in ("fp32", "int8")}
    valid_ids: dict[str, set[int]] = {"fp32": set(), "int8": set()}
    for model, predictions, ids in (
        ("fp32", fp32_predictions, fp_ids),
        ("int8", int8_predictions, int8_ids),
    ):
        for prediction_id in ids:
            prediction = predictions[prediction_id]
            image = images[str(prediction["image_id"])]
            ignored = any(
                obj.get("iscrowd", False)
                and int(obj["class_id"]) == int(prediction["class_id"])
                and bbox_iou(prediction["bbox_xyxy"], obj["bbox_xyxy"]) >= iou_threshold
                for obj in image["objects"]
            )
            if ignored:
                continue
            valid_ids[model].add(prediction_id)
            counts[model][_predicted_scale(prediction, image, scale_cfg)] += 1
    fp_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    int8_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for prediction_id in valid_ids["fp32"]:
        prediction = fp32_predictions[prediction_id]
        fp_by_key[(str(prediction["image_id"]), int(prediction["class_id"]))].append(
            prediction_id
        )
    for prediction_id in valid_ids["int8"]:
        prediction = int8_predictions[prediction_id]
        int8_by_key[(str(prediction["image_id"]), int(prediction["class_id"]))].append(
            prediction_id
        )
    paired_int8: set[int] = set()
    for key, fp_values in fp_by_key.items():
        fp_values.sort(
            key=lambda index: (-float(fp32_predictions[index]["score"]), index)
        )
        available = set(int8_by_key.get(key, []))
        for fp_id in fp_values:
            candidates = [
                (
                    bbox_iou(
                        fp32_predictions[fp_id]["bbox_xyxy"],
                        int8_predictions[int8_id]["bbox_xyxy"],
                    ),
                    int8_id,
                )
                for int8_id in available
            ]
            best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
            if best is not None and best[0] >= iou_threshold:
                available.remove(best[1])
                paired_int8.add(best[1])
    int8_only = valid_ids["int8"] - paired_int8
    int8_only_counts = {scale: 0 for scale in SCALES}
    for prediction_id in int8_only:
        prediction = int8_predictions[prediction_id]
        image = images[str(prediction["image_id"])]
        int8_only_counts[_predicted_scale(prediction, image, scale_cfg)] += 1
    n_images = len(ground_truth)
    return [
        {
            "scale": scale,
            "fp32_fp": counts["fp32"][scale],
            "int8_fp": counts["int8"][scale],
            "delta_fp": counts["int8"][scale] - counts["fp32"][scale],
            "int8_only_fp": int8_only_counts[scale],
            "int8_only_fraction": int8_only_counts[scale] / counts["int8"][scale]
            if counts["int8"][scale]
            else None,
            "fp32_fp_per_image": counts["fp32"][scale] / n_images if n_images else None,
            "int8_fp_per_image": counts["int8"][scale] / n_images if n_images else None,
        }
        for scale in SCALES
    ]


def pair_predictions(
    fp32_predictions: list[dict[str, Any]],
    int8_predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
    score_threshold: float,
    minimum_iou: float,
) -> list[dict[str, Any]]:
    """One-to-one FP32/INT8 prediction pairing for GT-independent displacement."""
    left, right = (
        _prediction_records(fp32_predictions, score_threshold),
        _prediction_records(int8_predictions, score_threshold),
    )
    images = {str(image["image_id"]): image for image in ground_truth}
    rows = []
    for key, fp_values in left.items():
        int8_values = right.get(key, [])
        available = set(range(len(int8_values)))
        for fp in fp_values:
            candidates = [
                (bbox_iou(fp["bbox_xyxy"], int8_values[index]["bbox_xyxy"]), index)
                for index in available
            ]
            best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
            if best is None or best[0] < minimum_iou:
                continue
            pair_iou, index = best
            available.remove(index)
            quant = int8_values[index]
            fx1, fy1, fx2, fy2 = map(float, fp["bbox_xyxy"])
            qx1, qy1, qx2, qy2 = map(float, quant["bbox_xyxy"])
            fw, fh, qw, qh = fx2 - fx1, fy2 - fy1, qx2 - qx1, qy2 - qy1
            dcx, dcy = (qx1 + qx2 - fx1 - fx2) / 2, (qy1 + qy2 - fy1 - fy2) / 2
            raw = math.hypot(dcx, dcy)
            normalizer = math.sqrt(max(fw * fh, 0.0))
            image = images[key[0]]
            rows.append(
                {
                    "image_id": fp["image_id"],
                    "class_id": fp["class_id"],
                    "scale": _predicted_scale(fp, image, scale_cfg),
                    "fp32_prediction_id": fp["prediction_id"],
                    "int8_prediction_id": quant["prediction_id"],
                    "pair_iou": pair_iou,
                    "fp32_score": fp["score"],
                    "int8_score": quant["score"],
                    "delta_score": quant["score"] - fp["score"],
                    "d_raw": raw,
                    "d_norm": raw / normalizer if normalizer > 0 else None,
                    "delta_w": qw - fw,
                    "delta_h": qh - fh,
                    "delta_w_relative": (qw - fw) / fw if fw > 0 else None,
                    "delta_h_relative": (qh - fh) / fh if fh > 0 else None,
                }
            )
    return rows


def decompose_failure_modes(
    rows: list[dict[str, Any]],
    int8_predictions: list[dict[str, Any]],
    score_threshold: float,
    iou_threshold: float,
    minimum_localization_iou: float,
) -> list[dict[str, Any]]:
    """Assign one frozen-priority mode to every FP32-success/INT8-failure."""
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction_id, prediction in enumerate(int8_predictions):
        by_image[str(prediction["image_id"])].append(
            {**prediction, "prediction_id": prediction_id}
        )
    output = []
    for row in rows:
        if row["transition"] != "10":
            continue
        candidates = by_image.get(str(row["image_id"]), [])
        same_class = [
            prediction
            for prediction in candidates
            if int(prediction["class_id"]) == int(row["class_id"])
        ]
        confidence = [
            prediction
            for prediction in same_class
            if float(prediction["score"]) < score_threshold
            and bbox_iou(prediction["bbox_xyxy"], row["gt_bbox_xyxy"]) >= iou_threshold
        ]
        localization = [
            prediction
            for prediction in same_class
            if float(prediction["score"]) >= score_threshold
            and minimum_localization_iou
            <= bbox_iou(prediction["bbox_xyxy"], row["gt_bbox_xyxy"])
            < iou_threshold
        ]
        class_failures = [
            prediction
            for prediction in candidates
            if int(prediction["class_id"]) != int(row["class_id"])
            and float(prediction["score"]) >= score_threshold
            and bbox_iou(prediction["bbox_xyxy"], row["gt_bbox_xyxy"]) >= iou_threshold
        ]
        if confidence:
            mode, evidence = (
                "confidence",
                max(confidence, key=lambda item: item["score"]),
            )
        elif localization:
            mode, evidence = (
                "localization",
                max(
                    localization,
                    key=lambda item: bbox_iou(item["bbox_xyxy"], row["gt_bbox_xyxy"]),
                ),
            )
        elif class_failures:
            mode, evidence = (
                "class",
                max(class_failures, key=lambda item: item["score"]),
            )
        else:
            mode, evidence = "complete_miss", None
        output.append(
            {
                "dataset": row["dataset"],
                "image_id": row["image_id"],
                "object_id": row["object_id"],
                "scale": row["scale"],
                "mode": mode,
                "evidence_prediction_id": None
                if evidence is None
                else evidence["prediction_id"],
            }
        )
    return output


def _prediction_index(
    predictions: list[dict[str, Any]], score_threshold: float
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction["score"]) >= score_threshold:
            index[(str(prediction["image_id"]), int(prediction["class_id"]))].append(
                prediction
            )
    return index


def match_predictions_to_gt(
    dataset_name: str,
    ground_truth: list[dict[str, Any]],
    fp32_predictions: list[dict[str, Any]],
    int8_predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
    score_threshold: float,
    match_iou_threshold: float,
) -> list[dict[str, Any]]:
    fp_index, int8_index = (
        _prediction_index(fp32_predictions, score_threshold),
        _prediction_index(int8_predictions, score_threshold),
    )
    rows = []
    for image in ground_truth:
        for obj in image["objects"]:
            if obj["iscrowd"]:
                continue
            key = (str(image["image_id"]), int(obj["class_id"]))

            def best(
                index: dict[tuple[str, int], list[dict[str, Any]]],
                gt_box: list[float] = obj["bbox_xyxy"],
                match_key: tuple[str, int] = key,
            ) -> tuple[float, float]:
                candidates = [
                    (
                        bbox_iou(gt_box, prediction["bbox_xyxy"]),
                        float(prediction["score"]),
                    )
                    for prediction in index.get(match_key, [])
                ]
                return max(
                    candidates,
                    key=lambda value: (value[0], value[1]),
                    default=(0.0, 0.0),
                )

            fp_iou, fp_score = best(fp_index)
            int8_iou, int8_score = best(int8_index)
            rows.append(
                {
                    "dataset": dataset_name,
                    "image_id": image["image_id"],
                    "object_id": obj["object_id"],
                    "class_id": obj["class_id"],
                    "scale": classify_object_scale(
                        obj["area"], image["width"], image["height"], scale_cfg
                    ),
                    "fp32_best_iou": fp_iou,
                    "int8_best_iou": int8_iou,
                    "fp32_score": fp_score,
                    "int8_score": int8_score,
                    "fp32_detected": fp_iou >= match_iou_threshold,
                    "int8_detected": int8_iou >= match_iou_threshold,
                }
            )
    return rows


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def confidence_shift_by_scale(matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        scale: _summary(
            [
                row["fp32_score"] - row["int8_score"]
                for row in matches
                if row["scale"] == scale
                and row["fp32_detected"]
                and row["int8_detected"]
            ]
        )
        for scale in ("small", "medium", "large")
    }


def localization_shift_by_scale(matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        scale: _summary(
            [
                row["fp32_best_iou"] - row["int8_best_iou"]
                for row in matches
                if row["scale"] == scale
                and row["fp32_detected"]
                and row["int8_detected"]
            ]
        )
        for scale in ("small", "medium", "large")
    }


def lost_detection_rate_by_scale(matches: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scale in ("small", "medium", "large"):
        fp_detected = [
            row for row in matches if row["scale"] == scale and row["fp32_detected"]
        ]
        lost = sum(not row["int8_detected"] for row in fp_detected)
        result[scale] = {
            "fp32_detected": len(fp_detected),
            "lost": lost,
            "rate": lost / len(fp_detected) if fp_detected else None,
        }
    return result


def summarize_diagnostics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "confidence_shift": confidence_shift_by_scale(matches),
        "localization_shift": localization_shift_by_scale(matches),
        "lost_detection_rate": lost_detection_rate_by_scale(matches),
    }
