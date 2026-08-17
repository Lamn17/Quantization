"""Native and unified COCO-style AP evaluation for all supported datasets."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dataset import classify_object_scale

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _interpolated_ap(tp: list[int], fp: list[int], n_gt: int) -> float:
    if n_gt == 0:
        raise ValueError("AP is undefined without ground truth")
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cumulative_tp / n_gt
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    return float(
        np.mean(
            [
                np.max(precision[recall >= level], initial=0.0)
                for level in np.linspace(0, 1, 101)
            ]
        )
    )


def _ap_for_scale(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
    target_scale: str | None,
) -> dict[str, float | None]:
    images = {str(image["image_id"]): image for image in ground_truth}
    valid_ids = set(images)
    predictions = [item for item in predictions if str(item["image_id"]) in valid_ids]
    gt_index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    class_ids = set()
    for image in ground_truth:
        for obj in image["objects"]:
            scale = classify_object_scale(
                float(obj["area"]), image["width"], image["height"], scale_cfg
            )
            enriched = {**obj, "scale": scale}
            gt_index[(str(image["image_id"]), int(obj["class_id"]))].append(enriched)
            if not obj["iscrowd"] and (target_scale is None or scale == target_scale):
                class_ids.add(int(obj["class_id"]))
    prediction_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        prediction_by_class[int(prediction["class_id"])].append(prediction)
    threshold_aps = []
    for threshold in IOU_THRESHOLDS:
        class_aps = []
        for class_id in sorted(class_ids):
            target_gt = {
                (key[0], index): obj
                for key, objects in gt_index.items()
                if key[1] == class_id
                for index, obj in enumerate(objects)
                if not obj["iscrowd"]
                and (target_scale is None or obj["scale"] == target_scale)
            }
            if not target_gt:
                continue
            matched: set[tuple[str, int]] = set()
            tp, fp = [], []
            for prediction in sorted(
                prediction_by_class.get(class_id, []),
                key=lambda item: -float(item["score"]),
            ):
                image_id = str(prediction["image_id"])
                candidates = gt_index.get((image_id, class_id), [])
                available = [
                    (index, obj, bbox_iou(prediction["bbox_xyxy"], obj["bbox_xyxy"]))
                    for index, obj in enumerate(candidates)
                    if (image_id, index) in target_gt
                    and (image_id, index) not in matched
                ]
                best = max(available, key=lambda item: item[2], default=None)
                if best and best[2] >= threshold:
                    matched.add((image_id, best[0]))
                    tp.append(1)
                    fp.append(0)
                    continue
                ignored_overlap = any(
                    (
                        obj["iscrowd"]
                        or (target_scale is not None and obj["scale"] != target_scale)
                    )
                    and bbox_iou(prediction["bbox_xyxy"], obj["bbox_xyxy"]) >= threshold
                    for obj in candidates
                )
                pred_area = max(
                    0.0, prediction["bbox_xyxy"][2] - prediction["bbox_xyxy"][0]
                ) * max(0.0, prediction["bbox_xyxy"][3] - prediction["bbox_xyxy"][1])
                pred_scale = classify_object_scale(
                    pred_area,
                    images[image_id]["width"],
                    images[image_id]["height"],
                    scale_cfg,
                )
                if ignored_overlap or (
                    target_scale is not None and pred_scale != target_scale
                ):
                    continue
                tp.append(0)
                fp.append(1)
            class_aps.append(_interpolated_ap(tp, fp, len(target_gt)))
        threshold_aps.append(float(np.mean(class_aps)) if class_aps else None)
    valid = [value for value in threshold_aps if value is not None]
    return {
        "AP": float(np.mean(valid)) if valid else None,
        "AP50": threshold_aps[0],
        "AP75": threshold_aps[5],
    }


def evaluate_generic_detection(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
) -> dict[str, Any]:
    return _ap_for_scale(ground_truth, predictions, scale_cfg, None)


def evaluate_scale_ap(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
) -> dict[str, Any]:
    overall = _ap_for_scale(ground_truth, predictions, scale_cfg, None)
    scales = {
        scale: _ap_for_scale(ground_truth, predictions, scale_cfg, scale)["AP"]
        for scale in ("small", "medium", "large")
    }
    return {
        "AP": overall["AP"],
        "AP50": overall["AP50"],
        "AP75": overall["AP75"],
        "AP_S": scales["small"],
        "AP_M": scales["medium"],
        "AP_L": scales["large"],
        "protocol": "unified_coco_style_101_point",
        "scale_bins": scale_cfg,
    }


def _evaluate_coco_native(
    dataset_cfg: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("COCO native evaluation requires pycocotools") from exc
    image_map = {
        str(image["image_id"]): index + 1 for index, image in enumerate(ground_truth)
    }
    names = dataset_cfg["_class_names"]
    coco_gt = {
        "info": {"description": "unified native COCO evaluation"},
        "licenses": [],
        "images": [
            {
                "id": image_map[str(image["image_id"])],
                "width": image["width"],
                "height": image["height"],
                "file_name": Path(image["image_path"]).name,
            }
            for image in ground_truth
        ],
        "categories": [
            {"id": index + 1, "name": name} for index, name in enumerate(names)
        ],
        "annotations": [],
    }
    annotation_id = 1
    for image in ground_truth:
        for obj in image["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            coco_gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_map[str(image["image_id"])],
                    "category_id": int(obj["class_id"]) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float(obj["area"]),
                    "iscrowd": int(obj["iscrowd"]),
                }
            )
            annotation_id += 1
    coco_predictions = [
        {
            "image_id": image_map[str(item["image_id"])],
            "category_id": int(item["class_id"]) + 1,
            "bbox": [
                item["bbox_xyxy"][0],
                item["bbox_xyxy"][1],
                item["bbox_xyxy"][2] - item["bbox_xyxy"][0],
                item["bbox_xyxy"][3] - item["bbox_xyxy"][1],
            ],
            "score": float(item["score"]),
        }
        for item in predictions
    ]
    with tempfile.TemporaryDirectory(prefix="lowbits_coco_eval_") as temporary:
        gt_path, pred_path = (
            Path(temporary) / "gt.json",
            Path(temporary) / "predictions.json",
        )
        gt_path.write_text(json.dumps(coco_gt))
        pred_path.write_text(json.dumps(coco_predictions))
        gt = COCO(str(gt_path))
        if coco_predictions:
            dt = gt.loadRes(str(pred_path))
        else:
            dt = COCO()
            dt.dataset = {
                "images": coco_gt["images"],
                "categories": coco_gt["categories"],
                "annotations": [],
            }
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = list(image_map.values())
        evaluator.evaluate()
        evaluator.accumulate()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            evaluator.summarize()
    names_out = (
        "AP",
        "AP50",
        "AP75",
        "AP_S",
        "AP_M",
        "AP_L",
        "AR1",
        "AR10",
        "AR100",
        "AR_S",
        "AR_M",
        "AR_L",
    )
    precision = evaluator.eval["precision"]  # IoU, recall, class, area, maxDet

    def precision_mean(iou_index: int | None, area_index: int) -> float | None:
        values = (
            precision[:, :, :, area_index, -1]
            if iou_index is None
            else precision[iou_index, :, :, area_index, -1]
        )
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    scale_iou = {
        scale: {
            "AP": precision_mean(None, area_index),
            "AP50": precision_mean(0, area_index),
            "AP75": precision_mean(5, area_index),
        }
        for scale, area_index in (("small", 1), ("medium", 2), ("large", 3))
    }
    return {
        **{name: float(value) for name, value in zip(names_out, evaluator.stats)},
        "protocol": "official_pycocotools",
        "scale_iou": scale_iou,
        "summary": output.getvalue(),
    }


def evaluate_official_coco(
    dataset_cfg: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply official COCOeval to the unified records of any supported dataset.

    This is the paper-metric path. Dataset-native protocols may additionally be
    reported, but the generic evaluator must not be used for headline numbers.
    """
    return _evaluate_coco_native(dataset_cfg, ground_truth, predictions)


def evaluate_native(
    dataset_cfg: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scale_cfg: dict[str, Any],
) -> dict[str, Any]:
    evaluator = dataset_cfg["native_evaluator"]
    if evaluator == "coco":
        return _evaluate_coco_native(dataset_cfg, ground_truth, predictions)
    metrics = evaluate_generic_detection(ground_truth, predictions, scale_cfg)
    metrics["protocol"] = (
        "generic_coco_style"
        if evaluator == "generic"
        else "generic_coco_style_fallback_not_official_visdrone"
    )
    return metrics


def compute_degradation(
    fp_metrics: dict[str, Any], quant_metrics: dict[str, Any], tolerance: float = 1e-6
) -> dict[str, Any]:
    fp, quant = (
        fp_metrics.get("scale", fp_metrics),
        quant_metrics.get("scale", quant_metrics),
    )
    output = {"warnings": []}
    for suffix, key in (("AP", "AP"), ("S", "AP_S"), ("M", "AP_M"), ("L", "AP_L")):
        if fp.get(key) is None or quant.get(key) is None:
            output[f"D_{suffix}"] = None
            output[f"RD_{suffix}"] = None
            continue
        output[f"D_{suffix}"] = float(fp[key]) - float(quant[key])
        output[f"RD_{suffix}"] = (
            output[f"D_{suffix}"] / (float(fp[key]) + 1e-12)
            if abs(float(fp[key])) > tolerance
            else None
        )
    output["SSG"] = (
        output["D_S"] - output["D_L"]
        if output["D_S"] is not None and output["D_L"] is not None
        else None
    )
    if output["D_L"] is None or abs(output["D_L"]) <= tolerance:
        output["SSR"] = None
        output["warnings"].append("SSR undefined because D_L is missing or near zero")
    else:
        output["SSR"] = (
            output["D_S"] / output["D_L"] if output["D_S"] is not None else None
        )
    return output
