from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibration import (
    sample_random,
    sample_scale_balanced,
    sample_small_rich,
    summarize_calibration,
)
from dataset import (
    build_dataset_metadata,
    dataset_statistics,
    load_dataset,
    load_dataset_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare unified TRAIN metadata and calibration lists"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = load_dataset_config(args.dataset)
    train = load_dataset(dataset_cfg, "train")
    metadata = build_dataset_metadata(train, config["scale_bins"])
    name = dataset_cfg["name"]
    repo = Path(__file__).resolve().parents[1]
    metadata_path = repo / "data" / "metadata" / f"{name}_train_stats.json"
    relocation_notes: list[str] = []

    def write_preserving(
        path: Path, content: str, allow_image_path_relocation: bool = False
    ) -> None:
        if path.exists() and not args.overwrite:
            existing_content = path.read_text(encoding="utf-8")
            if existing_content == content:
                return
            if allow_image_path_relocation:
                existing, candidate = json.loads(existing_content), json.loads(content)

                def scientific_fields(
                    rows: list[dict[str, object]],
                ) -> list[dict[str, object]]:
                    return [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "image_path"
                        }
                        for row in rows
                    ]

                if scientific_fields(existing) == scientific_fields(candidate):
                    relocation_notes.append(
                        f"Preserved {path}; only absolute image_path values changed with dataset relocation"
                    )
                    return
            raise FileExistsError(f"Artifact exists and differs: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    write_preserving(
        metadata_path,
        json.dumps(metadata, indent=2) + "\n",
        allow_image_path_relocation=True,
    )
    statistics_path = repo / "data" / "metadata" / f"{name}_dataset_statistics.json"
    write_preserving(
        statistics_path,
        json.dumps(dataset_statistics(train, config["scale_bins"]), indent=2) + "\n",
    )
    size = int(config["calibration"]["size"])
    output = repo / "data" / "calibration_lists"
    selections = []
    for seed in config["calibration"]["random_seeds"]:
        selections.append(
            (
                f"{name}_random_{size}_seed{seed}.txt",
                "random",
                int(seed),
                sample_random(metadata, size, int(seed)),
            )
        )
    selections.append(
        (
            f"{name}_small_rich_{size}.txt",
            "small_rich",
            None,
            sample_small_rich(metadata, size),
        )
    )
    balance_seed = int(config["runtime"].get("seed", 0))
    selections.append(
        (
            f"{name}_scale_balanced_{size}.txt",
            "scale_balanced",
            balance_seed,
            sample_scale_balanced(metadata, size, balance_seed),
        )
    )
    for filename, strategy, seed, ids in selections:
        path = output / filename
        serialized_ids = "".join(json.dumps(item) + "\n" for item in ids)
        write_preserving(path, serialized_ids)
        summary = {
            "dataset": name,
            "split": "train",
            "strategy": strategy,
            "seed": seed,
            **summarize_calibration(ids, metadata),
        }
        write_preserving(
            path.with_suffix(".summary.json"), json.dumps(summary, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "dataset": name,
                "train_images": len(train),
                "metadata": str(metadata_path),
                "statistics": str(statistics_path),
                "calibration_lists": len(selections),
                "notes": relocation_notes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
