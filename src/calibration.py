"""Dataset-independent calibration selection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _validate(metadata: list[dict[str, Any]], k: int) -> None:
    ids = [item["image_id"] for item in metadata]
    if k <= 0 or k > len(ids):
        raise ValueError(f"Calibration size must be in [1, {len(ids)}]")
    if len(ids) != len(set(map(str, ids))):
        raise ValueError("Metadata contains duplicate image IDs")


def sample_random(metadata: list[dict[str, Any]], k: int, seed: int) -> list[str | int]:
    _validate(metadata, k)
    return random.Random(seed).sample(
        [
            item["image_id"]
            for item in sorted(metadata, key=lambda row: str(row["image_id"]))
        ],
        k,
    )


def sample_small_rich(metadata: list[dict[str, Any]], k: int) -> list[str | int]:
    _validate(metadata, k)
    return [
        item["image_id"]
        for item in sorted(
            metadata, key=lambda row: (-int(row["n_small"]), str(row["image_id"]))
        )[:k]
    ]


def sample_scale_balanced(
    metadata: list[dict[str, Any]], k: int, seed: int
) -> list[str | int]:
    _validate(metadata, k)
    remaining = list(metadata)
    random.Random(seed).shuffle(remaining)
    totals = [0.0, 0.0, 0.0]
    fields = ("n_small", "n_medium", "n_large")
    selected = []
    for _ in range(k):

        def loss(
            item: dict[str, Any], current: list[float] = totals
        ) -> tuple[float, str]:
            candidate = [
                current[index] + float(item[field])
                for index, field in enumerate(fields)
            ]
            total = sum(candidate)
            imbalance = (
                sum((value / total - 1 / 3) ** 2 for value in candidate)
                if total
                else 1.0
            )
            return imbalance, str(item["image_id"])

        chosen = min(remaining, key=loss)
        remaining.remove(chosen)
        selected.append(chosen["image_id"])
        totals = [
            totals[index] + float(chosen[field]) for index, field in enumerate(fields)
        ]
    return selected


def summarize_calibration(
    ids: list[str | int], metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    lookup = {str(item["image_id"]): item for item in metadata}
    if (
        not ids
        or len(ids) != len(set(map(str, ids)))
        or any(str(item) not in lookup for item in ids)
    ):
        raise ValueError("Calibration IDs must be unique and present in TRAIN metadata")
    counts = {
        scale: sum(int(lookup[str(item)][f"n_{scale}"]) for item in ids)
        for scale in ("small", "medium", "large")
    }
    total = sum(counts.values())
    return {
        "size": len(ids),
        "n_small": counts["small"],
        "n_medium": counts["medium"],
        "n_large": counts["large"],
        "small_fraction": counts["small"] / total if total else 0.0,
    }


def save_calibration_list(ids: list[str | int], path: str | Path) -> None:
    if not ids or len(ids) != len(set(map(str, ids))):
        raise ValueError("Calibration list must contain unique IDs")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(item) + "\n" for item in ids), encoding="utf-8"
    )


def load_calibration_ids(
    path: str | Path, expected_size: int | None = None
) -> list[str | int]:
    ids = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(map(str, ids))):
        raise ValueError("Calibration list is empty or contains duplicate IDs")
    if expected_size is not None and len(ids) != expected_size:
        raise ValueError(
            f"Calibration list contains {len(ids)} IDs, expected {expected_size}"
        )
    return ids
