#!/usr/bin/env python3
"""Image-clustered bootstrap confidence intervals for scale-wise COCO AP.

Full re-evaluation per replicate is unaffordable, so the expensive step runs
once: ``COCOeval.evaluate()`` produces per-(category, area-range, image)
match records, and every replicate only re-runs ``COCOeval.accumulate()`` over
a resampled index into those records. The AP definition therefore stays exactly
the official pycocotools one.

Replicate resamples are keyed on (bootstrap seed, dataset, replicate) and never
on the run name, so FP32 and every INT8 seed see an identical image resample and
the per-replicate AP difference is properly paired.

Raw per-replicate AP vectors are written out, so paired intervals and any later
statistic can be recomputed without touching the GPU or re-evaluating.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_dataset, load_dataset_config

METRICS = ("AP", "AP_S", "AP_M", "AP_L")
AREA_INDEX = {"AP": 0, "AP_S": 1, "AP_M": 2, "AP_L": 3}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_evaluator(
    dataset_cfg: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
):
    """One COCOeval with evaluate() already run, restricted to maxDet=100."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_map = {
        str(image["image_id"]): index + 1 for index, image in enumerate(ground_truth)
    }
    names = dataset_cfg["_class_names"]
    coco_gt: dict[str, Any] = {
        "info": {"description": "bootstrap AP evaluation"},
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
    with tempfile.TemporaryDirectory(prefix="lowbits_bootstrap_ap_") as temporary:
        gt_path = Path(temporary) / "gt.json"
        prediction_path = Path(temporary) / "predictions.json"
        gt_path.write_text(json.dumps(coco_gt))
        prediction_path.write_text(json.dumps(coco_predictions))
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(gt_path))
            dt = gt.loadRes(str(prediction_path))
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.params.imgIds = list(image_map.values())
            # AP and its scale variants all read the largest maxDet slice, so
            # dropping the unused maxDet=1/10 slices leaves every reported
            # number identical while making accumulate() markedly cheaper.
            evaluator.params.maxDets = [100]
            evaluator.evaluate()
    return evaluator


def extract_ap(evaluator) -> dict[str, float | None]:
    precision = evaluator.eval["precision"]  # IoU, recall, class, area, maxDet

    def mean_precision(area_index: int) -> float | None:
        values = precision[:, :, :, area_index, -1]
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    return {metric: mean_precision(AREA_INDEX[metric]) for metric in METRICS}


def accumulate_on_resample(evaluator, base_records, base_ids, index: np.ndarray):
    """Re-accumulate over a bootstrap index without re-running evaluate()."""
    categories = len(evaluator.params.catIds)
    areas = len(evaluator.params.areaRng)
    images = len(base_ids)
    view = base_records.reshape(categories, areas, images)
    # The resampled list holds references to the original records, so this
    # costs pointers rather than a copy of the match data.
    evaluator.evalImgs = view[:, :, index].reshape(-1).tolist()
    resampled_ids = [base_ids[position] for position in index]
    evaluator.params.imgIds = resampled_ids
    evaluator._paramsEval.imgIds = resampled_ids
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
    return extract_ap(evaluator)


def interval(values: list[float]) -> dict[str, Any]:
    data = np.asarray([value for value in values if value is not None], dtype=float)
    if data.size == 0:
        return {"n": 0, "lower_95": None, "median": None, "upper_95": None, "std": None}
    return {
        "n": int(data.size),
        "lower_95": float(np.percentile(data, 2.5)),
        "median": float(np.percentile(data, 50)),
        "upper_95": float(np.percentile(data, 97.5)),
        "std": float(data.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument(
        "--dataset-config",
        action="append",
        required=True,
        help="Dataset config used by the source analysis, repeatable",
    )
    args = parser.parse_args()
    if not args.output_id.strip() or "/" in args.output_id or ".." in args.output_id:
        raise ValueError("--output-id must be a non-empty directory-safe name")
    analysis_root = Path(args.analysis_root).resolve()
    protocol = yaml.safe_load(
        (analysis_root / "protocol_resolved.yaml").read_text(encoding="utf-8")
    )["protocol"]
    config = yaml.safe_load(
        (analysis_root / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    output_root = Path(config["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    root = output_root / "bootstrap_ap" / args.output_id
    root.mkdir(parents=True, exist_ok=False)

    bootstrap_seed = int(protocol["bootstrap"]["seed"])
    seeds = [int(value) for value in config["calibration"]["random_seeds"]]
    run_names = ["fp32"] + [f"int8_random_s{seed}" for seed in seeds]
    started = time.time()
    manifest: dict[str, Any] = {
        "output_id": args.output_id,
        "analysis_root": str(analysis_root),
        "replicates": args.replicates,
        "bootstrap_seed": bootstrap_seed,
        "unit": "image",
        "runs": run_names,
        "notes": [
            (
                "COCOeval.evaluate() ran once per run; replicates re-ran "
                "accumulate() over a resampled image index."
            ),
            (
                "maxDets was restricted to [100]; AP, AP_S, AP_M and AP_L all "
                "read that slice, so the reported values are unchanged."
            ),
            (
                "Replicate resamples depend on (bootstrap seed, dataset, "
                "replicate) only, so FP32 and every INT8 seed share each resample."
            ),
        ],
        "datasets": {},
    }

    for dataset_index, dataset_path in enumerate(args.dataset_config):
        dataset_cfg = load_dataset_config(dataset_path)
        name = dataset_cfg["name"]
        print(f"[{time.time()-started:8.1f}s] {name}: loading ground truth", flush=True)
        records = load_dataset(dataset_cfg, "val")
        dataset_root = output_root / name
        image_ids = read_json(dataset_root / "fp32" / "evaluation_image_ids.json")
        lookup = {str(record["image_id"]): record for record in records}
        ground_truth = [lookup[str(image_id)] for image_id in image_ids]

        replicate_index: dict[int, np.ndarray] = {}
        per_run: dict[str, dict[str, list[float]]] = {}
        for run in run_names:
            prediction_path = dataset_root / run / "predictions.json"
            print(
                f"[{time.time()-started:8.1f}s] {name}/{run}: evaluate()", flush=True
            )
            predictions = read_json(prediction_path)
            evaluator = build_evaluator(dataset_cfg, ground_truth, predictions)
            del predictions
            base_ids = list(evaluator.params.imgIds)
            base_records = np.empty(len(evaluator.evalImgs), dtype=object)
            base_records[:] = evaluator.evalImgs
            if not replicate_index:
                for replicate in range(args.replicates):
                    rng = np.random.default_rng(
                        [bootstrap_seed, dataset_index, replicate]
                    )
                    replicate_index[replicate] = rng.integers(
                        0, len(base_ids), size=len(base_ids)
                    )
            collected: dict[str, list[float]] = {metric: [] for metric in METRICS}
            tick = time.time()
            for replicate in range(args.replicates):
                values = accumulate_on_resample(
                    evaluator, base_records, base_ids, replicate_index[replicate]
                )
                for metric in METRICS:
                    collected[metric].append(values[metric])
                if replicate == 19:
                    rate = (time.time() - tick) / 20
                    print(
                        f"[{time.time()-started:8.1f}s] {name}/{run}: "
                        f"{rate:.2f}s per replicate, "
                        f"projected {rate*args.replicates/60:.1f} min for this run",
                        flush=True,
                    )
            per_run[run] = collected
            write_json(
                root / name / f"replicates_{run}.json",
                {
                    "dataset": name,
                    "run": run,
                    "predictions_sha256": sha256(prediction_path),
                    "replicates": args.replicates,
                    "ap_by_replicate": collected,
                },
            )
            print(
                f"[{time.time()-started:8.1f}s] {name}/{run}: done in "
                f"{time.time()-tick:.1f}s",
                flush=True,
            )
            del evaluator, base_records

        rows = []
        for run in run_names:
            for metric in METRICS:
                rows.append(
                    {
                        "dataset": name,
                        "run": run,
                        "metric": metric,
                        "quantity": "ap",
                        **interval(per_run[run][metric]),
                    }
                )
        for run in run_names[1:]:
            for metric in METRICS:
                paired = [
                    fp - value
                    for fp, value in zip(
                        per_run["fp32"][metric], per_run[run][metric]
                    )
                    if fp is not None and value is not None
                ]
                rows.append(
                    {
                        "dataset": name,
                        "run": run,
                        "metric": metric,
                        "quantity": "delta_ap_vs_fp32_paired",
                        **interval(paired),
                    }
                )
        for metric in METRICS:
            pooled = []
            for replicate in range(args.replicates):
                fp = per_run["fp32"][metric][replicate]
                values = [
                    per_run[run][metric][replicate]
                    for run in run_names[1:]
                    if per_run[run][metric][replicate] is not None
                ]
                if fp is None or not values:
                    continue
                pooled.append(fp - statistics.mean(values))
            rows.append(
                {
                    "dataset": name,
                    "run": "int8_random_seed_mean",
                    "metric": metric,
                    "quantity": "delta_ap_vs_fp32_paired",
                    **interval(pooled),
                }
            )
        write_json(root / name / "intervals.json", rows)
        manifest["datasets"][name] = {
            "dataset_config": str(Path(dataset_path).resolve()),
            "dataset_config_sha256": sha256(Path(dataset_path).resolve()),
            "images": len(ground_truth),
            "interval_rows": len(rows),
        }
        write_json(root / "bootstrap_ap_manifest.json", manifest)
        print(f"[{time.time()-started:8.1f}s] {name}: intervals written", flush=True)

    manifest["elapsed_seconds"] = time.time() - started
    write_json(root / "bootstrap_ap_manifest.json", manifest)
    print(json.dumps({"bootstrap_ap_root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
