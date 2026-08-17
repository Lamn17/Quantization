from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibration import load_calibration_ids
from dataset import load_dataset, load_dataset_config
from evaluation import evaluate_native, evaluate_official_coco, evaluate_scale_ap
from inference import load_model, run_inference, validate_model_classes
from quantization import build_int8_model, build_tensorrt_fp16_control


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def seed_everything(seed: int, deterministic: bool) -> list[str]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    warnings = []
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        warnings.append(
            "PyTorch unavailable during seeding; inference will also be unavailable"
        )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dataset/precision experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--precision", choices=["fp32", "trt_fp16", "int8"], required=True
    )
    parser.add_argument(
        "--calibration",
        choices=["random", "small_rich", "scale_balanced"],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--run-suffix",
        help="Append a safe suffix for reproducibility/control runs without overwriting",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = load_dataset_config(args.dataset)
    name = dataset_cfg["name"]
    run_name = (
        args.precision
        if args.precision in {"fp32", "trt_fp16"}
        else (
            f"int8_random_s{args.seed}"
            if args.calibration == "random"
            else f"int8_{args.calibration}"
        )
    )
    if args.run_suffix:
        if any(token in args.run_suffix for token in ("/", "\\", "..")):
            raise ValueError("--run-suffix must be directory-safe")
        run_name = f"{run_name}_{args.run_suffix}"
    output_root = Path(config["paths"]["output_root"]).expanduser().resolve()
    run_dir = output_root / name / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run exists and will not be overwritten: {run_dir}")
        shutil.rmtree(run_dir)
    (run_dir / "artifacts").mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    warnings = seed_everything(
        args.seed, bool(config["runtime"].get("deterministic", True))
    )
    checkpoint = dataset_cfg.get("model_override", {}).get(
        "checkpoint", config["paths"]["checkpoint"]
    )
    checkpoint = Path(checkpoint).expanduser().resolve()
    run_info = {
        "status": "running",
        "dataset": name,
        "dataset_format": dataset_cfg["format"],
        "run_name": run_name,
        "precision": args.precision,
        "checkpoint": str(checkpoint),
        "calibration": args.calibration if args.precision == "int8" else None,
        "seed": args.seed,
        "max_images": args.max_images,
        "started_at": started,
        "python": platform.python_version(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "software_versions": {
            name: package_version(name)
            for name in ("ultralytics", "torch", "tensorrt", "pycocotools", "numpy")
        },
        "warnings": warnings,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    resolved = {
        "global": config,
        "dataset": {
            key: value for key, value in dataset_cfg.items() if not key.startswith("_")
        },
        "command": vars(args),
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    try:
        val_records = load_dataset(dataset_cfg, "val")
        if args.max_images is not None:
            if args.max_images <= 0:
                raise ValueError("--max-images must be positive")
            val_records = val_records[: args.max_images]
        (run_dir / "evaluation_image_ids.json").write_text(
            json.dumps([item["image_id"] for item in val_records], indent=2) + "\n",
            encoding="utf-8",
        )
        base_model = load_model(checkpoint)
        validate_model_classes(base_model, dataset_cfg)
        run_info["checkpoint_sha256"] = file_sha256(checkpoint)
        model_path = checkpoint
        model = base_model
        if args.precision == "trt_fp16":
            model_path = build_tensorrt_fp16_control(
                checkpoint,
                run_dir / "artifacts",
                int(config["model"]["imgsz"]),
                str(config["runtime"]["device"]),
                float(config["quantization"].get("workspace_gb", 4)),
            )
            run_info["engine"] = str(model_path)
            run_info["engine_sha256"] = file_sha256(model_path)
            model = load_model(model_path)
            validate_model_classes(model, dataset_cfg)
        if args.precision == "int8":
            train_records = load_dataset(dataset_cfg, "train")
            size = int(config["calibration"]["size"])
            repo = Path(__file__).resolve().parents[1]
            filename = (
                f"{name}_random_{size}_seed{args.seed}.txt"
                if args.calibration == "random"
                else f"{name}_{args.calibration}_{size}.txt"
            )
            calibration_path = repo / "data" / "calibration_lists" / filename
            ids = load_calibration_ids(calibration_path, size)
            (run_dir / "calibration_ids.txt").write_text(
                calibration_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            model_path = build_int8_model(
                checkpoint,
                calibration_path,
                train_records,
                dataset_cfg["_class_names"],
                run_dir / "artifacts",
                int(config["model"]["imgsz"]),
                str(config["runtime"]["device"]),
                float(config["quantization"].get("workspace_gb", 4)),
                size,
            )
            run_info["calibration_list"] = str(calibration_path)
            run_info["calibration_size"] = len(ids)
            run_info["engine"] = str(model_path)
            run_info["engine_sha256"] = file_sha256(model_path)
            model = load_model(model_path)
            validate_model_classes(model, dataset_cfg)
        rejected_predictions = []
        predictions = run_inference(
            model,
            val_records,
            int(config["model"]["imgsz"]),
            str(config["runtime"]["device"]),
            float(config["evaluation"]["conf_threshold_export"]),
            float(config["evaluation"]["nms_iou"]),
            int(config["evaluation"]["max_detections"]),
            rejected_predictions,
        )
        (run_dir / "predictions.json").write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
        if rejected_predictions:
            (run_dir / "rejected_predictions.json").write_text(
                json.dumps(rejected_predictions, indent=2) + "\n", encoding="utf-8"
            )
            run_info["warnings"].append(
                f"Excluded {len(rejected_predictions)} predictions with negative bbox extent; raw values are preserved in rejected_predictions.json"
            )
        native_metrics = evaluate_native(
            dataset_cfg, val_records, predictions, config["scale_bins"]
        )
        scale_exploratory = evaluate_scale_ap(
            val_records, predictions, config["scale_bins"]
        )
        metrics = {
            "paper_official_coco": (
                native_metrics
                if dataset_cfg["native_evaluator"] == "coco"
                else evaluate_official_coco(dataset_cfg, val_records, predictions)
            ),
            "native": native_metrics,
            "scale_exploratory": scale_exploratory,
            "scale": scale_exploratory,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        run_info.update(
            {
                "status": "completed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_images": len(val_records),
                "predictions": len(predictions),
                "rejected_predictions": len(rejected_predictions),
            }
        )
    except Exception as exc:
        run_info.update(
            {
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        (run_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
        )
        raise
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(run_dir)


if __name__ == "__main__":
    main()
