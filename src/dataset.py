"""Simple adapters from COCO, YOLO TXT, and VisDrone to one unified format."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw.get("dataset"), dict):
        raise TypeError("Dataset config must contain a 'dataset' mapping")
    config = raw["dataset"]
    required = {"name", "format", "root", "train", "val", "native_evaluator"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Dataset config is missing keys: {missing}")
    if config["format"] not in {"coco", "yolo", "visdrone"}:
        raise ValueError("dataset.format must be coco, yolo, or visdrone")
    config["_config_path"] = str(config_path)
    return config


def _split_config(
    dataset_cfg: dict[str, Any], split: str
) -> tuple[Path, dict[str, Any]]:
    if split not in {"train", "val", "test"} or split not in dataset_cfg:
        raise ValueError(f"Dataset has no supported split {split!r}")
    root = Path(dataset_cfg["root"]).expanduser()
    if not root.is_absolute():
        config_parent = Path(dataset_cfg.get("_config_path", Path.cwd())).parent
        root = config_parent / root
    root = root.resolve()
    split_cfg = dataset_cfg[split]
    image_dir = root / split_cfg["images"]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return root, split_cfg


def _images(image_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found under {image_dir}")
    return paths


def _record(
    image_id: str | int,
    image_path: Path,
    width: int,
    height: int,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    absolute_image_path = (
        image_path if image_path.is_absolute() else image_path.resolve()
    )
    return {
        "image_id": image_id,
        "image_path": str(absolute_image_path),
        "width": int(width),
        "height": int(height),
        "objects": objects,
    }


def load_coco_dataset(dataset_cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    root, split_cfg = _split_config(dataset_cfg, split)
    annotation_path = root / split_cfg["annotations"]
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotations do not exist: {annotation_path}")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
    category_to_class = {
        int(category["id"]): index for index, category in enumerate(categories)
    }
    class_names = [str(category["name"]) for category in categories]
    if not class_names:
        raise ValueError("COCO annotations contain no categories")
    dataset_cfg["_class_names"] = class_names
    dataset_cfg["_category_ids"] = [int(category["id"]) for category in categories]
    by_image: dict[int, list[dict[str, Any]]] = {
        int(image["id"]): [] for image in data.get("images", [])
    }
    for annotation in data.get("annotations", []):
        image_id, category_id = (
            int(annotation["image_id"]),
            int(annotation["category_id"]),
        )
        if image_id not in by_image or category_id not in category_to_class:
            raise ValueError(
                f"Invalid COCO annotation reference: {annotation.get('id')}"
            )
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if width < 0 or height < 0:
            raise ValueError(f"Negative COCO bbox extent: {annotation.get('id')}")
        class_id = category_to_class[category_id]
        by_image[image_id].append(
            {
                "object_id": int(annotation["id"]),
                "class_id": class_id,
                "class_name": class_names[class_id],
                "bbox_xyxy": [x, y, x + width, y + height],
                "area": float(annotation.get("area", width * height)),
                "iscrowd": bool(annotation.get("iscrowd", 0)),
            }
        )
    image_dir = root / split_cfg["images"]
    available_images = {
        entry.name for entry in os.scandir(image_dir) if entry.is_file()
    }
    records = []
    for image in sorted(data.get("images", []), key=lambda item: int(item["id"])):
        path = image_dir / image["file_name"]
        if image["file_name"] not in available_images:
            raise FileNotFoundError(f"COCO image does not exist: {path}")
        image_id = int(image["id"])
        records.append(
            _record(image_id, path, image["width"], image["height"], by_image[image_id])
        )
    return records


def _configured_names(dataset_cfg: dict[str, Any]) -> list[str]:
    names = dataset_cfg.get("class_names")
    if not isinstance(names, list) or not names:
        raise ValueError("YOLO/VisDrone dataset requires a non-empty class_names list")
    return [str(name) for name in names]


def load_yolo_dataset(dataset_cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    root, split_cfg = _split_config(dataset_cfg, split)
    names = _configured_names(dataset_cfg)
    dataset_cfg["_class_names"] = names
    image_dir, label_dir = root / split_cfg["images"], root / split_cfg["labels"]
    if not label_dir.is_dir():
        raise FileNotFoundError(f"YOLO label directory does not exist: {label_dir}")
    records = []
    for image_path in _images(image_dir):
        relative = image_path.relative_to(image_dir)
        image_id = relative.with_suffix("").as_posix()
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = label_dir / relative.with_suffix(".txt")
        objects = []
        if label_path.is_file():
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 5:
                    raise ValueError(
                        f"Expected YOLO detection row with 5 values: {label_path}:{line_number}"
                    )
                class_id = int(values[0])
                cx, cy, box_width, box_height = map(float, values[1:])
                if (
                    not 0 <= class_id < len(names)
                    or min(cx, cy, box_width, box_height) < 0
                    or max(cx, cy, box_width, box_height) > 1
                ):
                    raise ValueError(f"Invalid YOLO label: {label_path}:{line_number}")
                x1, y1 = (cx - box_width / 2) * width, (cy - box_height / 2) * height
                x2, y2 = (cx + box_width / 2) * width, (cy + box_height / 2) * height
                objects.append(
                    {
                        "object_id": f"{image_id}:{line_number}",
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "area": box_width * width * box_height * height,
                        "iscrowd": False,
                    }
                )
        records.append(_record(image_id, image_path, width, height, objects))
    return records


def load_visdrone_dataset(
    dataset_cfg: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    root, split_cfg = _split_config(dataset_cfg, split)
    names = (
        _configured_names(dataset_cfg)
        if dataset_cfg.get("class_names")
        else VISDRONE_CLASS_NAMES
    )
    if len(names) != 10:
        raise ValueError("VisDrone detection requires exactly 10 class names")
    dataset_cfg["_class_names"] = names
    image_dir, annotation_dir = (
        root / split_cfg["images"],
        root / split_cfg["annotations"],
    )
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"VisDrone annotations do not exist: {annotation_dir}")
    records = []
    for image_path in _images(image_dir):
        relative = image_path.relative_to(image_dir)
        image_id = relative.with_suffix("").as_posix()
        with Image.open(image_path) as image:
            width, height = image.size
        annotation_path = annotation_dir / relative.with_suffix(".txt")
        objects = []
        if not annotation_path.is_file():
            raise FileNotFoundError(f"VisDrone annotation missing: {annotation_path}")
        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) < 8:
                raise ValueError(
                    f"Invalid VisDrone row: {annotation_path}:{line_number}"
                )
            x, y, box_width, box_height = map(float, values[:4])
            category = int(values[5])
            if box_width < 0 or box_height < 0:
                raise ValueError(
                    f"Invalid VisDrone object: {annotation_path}:{line_number}"
                )
            if category == 0:
                # VisDrone ignored regions are category-agnostic.  Represent each
                # region once per scored class so COCOeval and the paired false-
                # positive accounting can suppress overlapping predictions of
                # any class without adding the region to scored GT counts.
                for class_id, class_name in enumerate(names):
                    objects.append(
                        {
                            "object_id": f"{image_id}:{line_number}:ignore:{class_id}",
                            "class_id": class_id,
                            "class_name": class_name,
                            "bbox_xyxy": [x, y, x + box_width, y + box_height],
                            "area": box_width * box_height,
                            "iscrowd": True,
                            "ignore_region": True,
                        }
                    )
                continue
            if category == 11:
                continue  # "others" is not a scored VisDrone category
            if not 1 <= category <= 10:
                raise ValueError(
                    f"Invalid VisDrone object: {annotation_path}:{line_number}"
                )
            class_id = category - 1
            objects.append(
                {
                    "object_id": f"{image_id}:{line_number}",
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "bbox_xyxy": [x, y, x + box_width, y + box_height],
                    "area": box_width * box_height,
                    "iscrowd": False,
                }
            )
        records.append(_record(image_id, image_path, width, height, objects))
    return records


def load_dataset(dataset_cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    loaders = {
        "coco": load_coco_dataset,
        "yolo": load_yolo_dataset,
        "visdrone": load_visdrone_dataset,
    }
    return loaders[dataset_cfg["format"]](dataset_cfg, split)


def classify_object_scale(
    bbox_area: float, image_width: int, image_height: int, scale_cfg: dict[str, Any]
) -> str:
    if bbox_area < 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("Area and image dimensions must be valid")
    mode = scale_cfg.get("mode", "absolute_pixels")
    if mode == "absolute_pixels":
        value, small, medium = (
            bbox_area,
            float(scale_cfg["small_max_area"]),
            float(scale_cfg["medium_max_area"]),
        )
    elif mode == "relative_area":
        value, small, medium = (
            bbox_area / (image_width * image_height),
            float(scale_cfg["small_max_ratio"]),
            float(scale_cfg["medium_max_ratio"]),
        )
    else:
        raise ValueError("scale_bins.mode must be absolute_pixels or relative_area")
    if not 0 <= small < medium:
        raise ValueError("Scale thresholds must satisfy 0 <= small < medium")
    return "small" if value < small else ("medium" if value < medium else "large")


def build_dataset_metadata(
    dataset: list[dict[str, Any]], scale_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = []
    for image in dataset:
        counts = {"small": 0, "medium": 0, "large": 0}
        for obj in image["objects"]:
            if not obj["iscrowd"]:
                counts[
                    classify_object_scale(
                        obj["area"], image["width"], image["height"], scale_cfg
                    )
                ] += 1
        total = sum(counts.values())
        metadata.append(
            {
                "image_id": image["image_id"],
                "image_path": image["image_path"],
                "width": image["width"],
                "height": image["height"],
                "n_objects": total,
                "n_small": counts["small"],
                "n_medium": counts["medium"],
                "n_large": counts["large"],
                "small_ratio": counts["small"] / total if total else 0.0,
            }
        )
    return sorted(metadata, key=lambda item: str(item["image_id"]))


def dataset_statistics(
    dataset: list[dict[str, Any]], scale_cfg: dict[str, Any]
) -> dict[str, Any]:
    metadata = build_dataset_metadata(dataset, scale_cfg)
    areas, ratios = [], []
    for image in dataset:
        for obj in image["objects"]:
            if not obj["iscrowd"]:
                areas.append(float(obj["area"]))
                ratios.append(float(obj["area"]) / (image["width"] * image["height"]))
    counts = {
        scale: sum(item[f"n_{scale}"] for item in metadata)
        for scale in ("small", "medium", "large")
    }
    total = sum(counts.values())
    return {
        "number_of_images": len(dataset),
        "number_of_objects": total,
        "small_count": counts["small"],
        "medium_count": counts["medium"],
        "large_count": counts["large"],
        "small_percent": 100 * counts["small"] / total if total else 0.0,
        "median_bbox_area": statistics.median(areas) if areas else None,
        "median_relative_bbox_area": statistics.median(ratios) if ratios else None,
        "scale_bins": scale_cfg,
    }


def save_metadata(metadata: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_metadata(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
