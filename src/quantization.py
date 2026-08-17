"""Dataset-independent exact-subset TensorRT INT8 integration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from calibration import load_calibration_ids


def _engine_payload(engine_path: Path) -> tuple[bytes, dict[str, Any] | None]:
    """Return raw TensorRT bytes and optional Ultralytics prefix metadata."""
    content = engine_path.read_bytes()
    if len(content) < 5:
        return content, None
    metadata_length = int.from_bytes(content[:4], byteorder="little")
    payload_offset = 4 + metadata_length
    if metadata_length <= 0 or payload_offset >= len(content):
        return content, None
    try:
        metadata = json.loads(content[4:payload_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return content, None
    if not isinstance(metadata, dict):
        return content, None
    return content[payload_offset:], metadata


def _stage_checkpoint(checkpoint_path: Path, artifacts: Path) -> Path:
    """Keep exporter side effects inside the immutable run directory."""
    stage = artifacts / "export_source"
    stage.mkdir(parents=True, exist_ok=False)
    staged = stage / checkpoint_path.name
    shutil.copy2(checkpoint_path, staged)
    return staged


def _inspect_engine(engine_path: Path, artifacts: Path) -> dict[str, Any]:
    """Persist TensorRT layer information using Python, then ``trtexec``."""
    target = artifacts / "engine_inspector.json"
    payload, metadata = _engine_payload(engine_path)
    metadata_path: Path | None = None
    if metadata is not None:
        metadata_path = artifacts / "ultralytics_engine_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    python_error: str | None = None
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the exported engine")
        inspector = engine.create_engine_inspector()
        raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        parsed = json.loads(raw)
        target.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "method": "python_api",
            "path": str(target),
            "ultralytics_metadata": str(metadata_path) if metadata_path else None,
        }
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        python_error = f"{type(exc).__name__}: {exc}"

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        result = {
            "status": "unavailable",
            "python_api_error": python_error,
            "trtexec_error": "trtexec executable not found on PATH",
        }
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    exported = artifacts / "trtexec_layer_info.json"
    log_path = artifacts / "trtexec_layer_info.log"
    try:
        with tempfile.TemporaryDirectory(prefix="lowbits_engine_payload_") as temporary:
            raw_engine = Path(temporary) / engine_path.name
            raw_engine.write_bytes(payload)
            command = [
                trtexec,
                f"--loadEngine={raw_engine}",
                "--skipInference",
                "--dumpLayerInfo",
                f"--exportLayerInfo={exported}",
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        log_path.write_text(
            completed.stdout + "\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not exported.is_file():
            raise RuntimeError(
                f"trtexec returned {completed.returncode}; see {log_path}"
            )
        parsed = json.loads(exported.read_text(encoding="utf-8"))
        target.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "method": "trtexec",
            "path": str(target),
            "log": str(log_path),
            "ultralytics_metadata": str(metadata_path) if metadata_path else None,
        }
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as exc:
        result = {
            "status": "unavailable",
            "python_api_error": python_error,
            "trtexec_error": f"{type(exc).__name__}: {exc}",
            "log": str(log_path) if log_path.is_file() else None,
        }
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def resolve_calibration_images(
    ids: list[str | int], train_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {str(record["image_id"]): record for record in train_records}
    if not ids or len(ids) != len(set(map(str, ids))):
        raise ValueError("Calibration IDs must be non-empty and unique")
    missing = [item for item in ids if str(item) not in lookup]
    if missing:
        raise ValueError(f"Calibration IDs are absent from TRAIN: {missing[:10]}")
    return [lookup[str(item)] for item in ids]


def _exact_dataset(
    records: list[dict[str, Any]], destination: Path, class_names: list[str]
) -> Path:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Calibration adapter exists: {destination}")
    image_dir, label_dir = (
        destination / "images" / "train",
        destination / "labels" / "train",
    )
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest = []
    for index, record in enumerate(records):
        source = Path(record["image_path"])
        target = image_dir / f"{index:06d}{source.suffix.lower()}"
        if not source.is_file():
            raise FileNotFoundError(f"Calibration image missing: {source}")
        try:
            target.symlink_to(source.resolve())
            mode = "symlink"
        except OSError:
            shutil.copy2(source, target)
            mode = "copy"
        (label_dir / f"{target.stem}.txt").write_text("", encoding="utf-8")
        manifest.append(
            {
                "image_id": record["image_id"],
                "source": str(source.resolve()),
                "adapter_path": str(target),
                "mode": mode,
            }
        )
    (destination / "manifest.json").write_text(
        json.dumps({"count": len(manifest), "records": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    yaml_path = destination / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/train",
                "names": {index: name for index, name in enumerate(class_names)},
            }
        ),
        encoding="utf-8",
    )
    if len(list(image_dir.iterdir())) != len(records):
        raise RuntimeError("Exact calibration adapter image count mismatch")
    return yaml_path


def run_int8_calibration(
    checkpoint: str | Path,
    calibration_records: list[dict[str, Any]],
    class_names: list[str],
    artifacts_dir: str | Path,
    imgsz: int,
    device: str,
    workspace_gb: float = 4,
) -> Path:
    os.environ["YOLO_AUTOINSTALL"] = "false"
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Local checkpoint missing: {checkpoint_path}")
    try:
        import tensorrt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is unavailable; real INT8 cannot run and will not be simulated"
        ) from exc
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for TensorRT export") from exc
    artifacts = Path(artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = _stage_checkpoint(checkpoint_path, artifacts)
    data_yaml = _exact_dataset(
        calibration_records, artifacts / "calibration_dataset", class_names
    )
    exported = YOLO(str(staged_checkpoint), task="detect").export(
        format="engine",
        int8=True,
        data=str(data_yaml),
        fraction=1.0,
        imgsz=imgsz,
        batch=1,
        device=device,
        workspace=workspace_gb,
        verbose=False,
    )
    source_engine = Path(str(exported)).resolve()
    if not source_engine.is_file():
        raise RuntimeError(f"TensorRT export did not create an engine: {source_engine}")
    engine = artifacts / source_engine.name
    if engine.resolve() != source_engine:
        shutil.copy2(source_engine, engine)
    settings = {
        "backend": "TensorRT",
        "precision": "INT8",
        "exact_calibration_subset": True,
        "calibration_fraction": 1.0,
        "calibration_size": len(calibration_records),
        "imgsz": imgsz,
        "device": device,
        "workspace_gb": workspace_gb,
        "manifest": str(artifacts / "calibration_dataset" / "manifest.json"),
        "weight_quantization_scheme": "backend_managed_not_verified",
        "activation_calibration_algorithm": "ultralytics_tensorrt_default_not_verified",
        "engine_inspection": _inspect_engine(engine, artifacts),
    }
    (artifacts / "backend_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    return engine.resolve()


def build_tensorrt_fp16_control(
    checkpoint: str | Path,
    artifacts_dir: str | Path,
    imgsz: int,
    device: str,
    workspace_gb: float = 4,
) -> Path:
    """Export a same-backend FP16 control without calibration data."""
    os.environ["YOLO_AUTOINSTALL"] = "false"
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Local checkpoint missing: {checkpoint_path}")
    try:
        import tensorrt  # noqa: F401
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT and Ultralytics are required for FP16 control"
        ) from exc
    artifacts = Path(artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = _stage_checkpoint(checkpoint_path, artifacts)
    exported = YOLO(str(staged_checkpoint), task="detect").export(
        format="engine",
        half=True,
        int8=False,
        imgsz=imgsz,
        batch=1,
        device=device,
        workspace=workspace_gb,
        verbose=False,
    )
    source_engine = Path(str(exported)).resolve()
    if not source_engine.is_file():
        raise RuntimeError(f"TensorRT export did not create an engine: {source_engine}")
    engine = artifacts / f"{checkpoint_path.stem}.fp16.engine"
    if engine.resolve() != source_engine:
        shutil.copy2(source_engine, engine)
    (artifacts / "backend_settings.json").write_text(
        json.dumps(
            {
                "backend": "TensorRT",
                "precision": "FP16",
                "calibration": None,
                "imgsz": imgsz,
                "device": device,
                "workspace_gb": workspace_gb,
                "engine_inspection": _inspect_engine(engine, artifacts),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return engine.resolve()


def build_int8_model(
    checkpoint: str | Path,
    calibration_ids_path: str | Path,
    train_records: list[dict[str, Any]],
    class_names: list[str],
    artifacts_dir: str | Path,
    imgsz: int,
    device: str,
    workspace_gb: float = 4,
    expected_size: int | None = None,
) -> Path:
    ids = load_calibration_ids(calibration_ids_path, expected_size)
    records = resolve_calibration_images(ids, train_records)
    return run_int8_calibration(
        checkpoint, records, class_names, artifacts_dir, imgsz, device, workspace_gb
    )
