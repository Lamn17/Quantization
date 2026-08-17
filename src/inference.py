"""Dataset-independent local YOLO inference returning unified predictions."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm


def load_model(checkpoint: str | Path) -> Any:
    os.environ["YOLO_AUTOINSTALL"] = "false"
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Local checkpoint/engine missing; no download attempted: {path}"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Install ultralytics from a local/approved package source before inference"
        ) from exc
    return YOLO(str(path), task="detect")


def validate_model_classes(model: Any, dataset_cfg: dict[str, Any]) -> None:
    expected = dataset_cfg.get("_class_names") or dataset_cfg.get("class_names")
    if expected == "auto" or not expected:
        raise ValueError("Load the dataset before validating model classes")
    actual = getattr(model, "names", None)
    if isinstance(actual, dict):
        actual = [actual[index] for index in sorted(actual)]
    if actual is None and getattr(model, "model", None) is not None:
        actual = getattr(model.model, "names", None)
    if isinstance(actual, dict):
        actual = [actual[index] for index in sorted(actual)]
    override_count = dataset_cfg.get("model_override", {}).get("num_classes")
    expected_count = (
        int(override_count) if override_count is not None else len(expected)
    )
    if actual is not None and len(actual) != expected_count:
        raise ValueError(
            f"Checkpoint exposes {len(actual)} classes but dataset requires {expected_count}; use a dataset-specific checkpoint"
        )


def run_inference(
    model: Any,
    image_records: list[dict[str, Any]],
    imgsz: int,
    device: str,
    conf_threshold: float,
    nms_iou: float,
    max_detections: int,
    rejected_predictions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    predictions = []
    for image in tqdm(image_records, desc="inference"):
        result = model.predict(
            source=image["image_path"],
            imgsz=imgsz,
            device=device,
            conf=conf_threshold,
            iou=nms_iou,
            max_det=max_detections,
            verbose=False,
            save=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            continue
        for xyxy, score, class_value in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        ):
            values = [float(value) for value in xyxy]
            confidence = float(score)
            class_id = int(class_value)
            if (
                len(values) != 4
                or not all(math.isfinite(value) for value in [*values, confidence])
                or not 0 <= confidence <= 1
            ):
                raise ValueError(
                    f"Model produced invalid prediction for image {image['image_id']}"
                )
            if values[2] < values[0] or values[3] < values[1]:
                if rejected_predictions is None:
                    raise ValueError("Model produced bbox with negative extent")
                rejected_predictions.append(
                    {
                        "image_id": image["image_id"],
                        "class_id": class_id,
                        "bbox_xyxy": values,
                        "score": confidence,
                        "reason": "negative_bbox_extent",
                    }
                )
                continue
            predictions.append(
                {
                    "image_id": image["image_id"],
                    "class_id": class_id,
                    "bbox_xyxy": values,
                    "score": confidence,
                }
            )
    return predictions
