# Measuring Scale-Specific Robustness of Quantized Object Detectors

Research code for measuring how post-training INT8 TensorRT quantization changes
object-detection performance across object scales and datasets. The benchmark is
offline-first: it never downloads datasets or checkpoints, and it does not train
or modify the supplied model.

The main paper metrics are computed with official `pycocotools.COCOeval`. Raw
predictions, resolved configuration, calibration image IDs, engines, and
analysis tables are retained so every reported number can be traced to an
artifact.

## Scope and guarantees

- Supported annotations: COCO JSON, YOLO TXT, and native VisDrone TXT.
- Supported experiment modes: FP32, TensorRT FP16 control, and real TensorRT
  INT8 post-training quantization.
- No training, QAT, pruning, distillation, simulated INT4, or automatic data
  download.
- Calibration always comes from the training split; COCO `val2017` is not used
  as primary calibration data.
- Existing experiment and analysis directories are rejected by default. Use a
  new run/analysis ID instead of overwriting previous evidence.
- Failed or missing runs are recorded in manifests; they are never silently
  omitted from a paper analysis.

## Repository layout

```text
.
├── config.yaml                     # runtime, calibration, scale-bin settings
├── datasets/
│   ├── coco.yaml                   # COCO 2017 configuration
│   ├── tt100k.yaml                 # TT100K YOLO configuration
│   └── visdrone.yaml               # VisDrone annotations configuration
├── protocols/
│   └── framing.yaml                # frozen paper-analysis protocol
├── scripts/
│   ├── prepare_data.py             # build train metadata and calibration lists
│   ├── run_experiment.py           # run FP32, TensorRT FP16, or TensorRT INT8
│   ├── analyze.py                  # official AP, degradation, and diagnostics
│   ├── report.py                   # paper tables, figures, and source CSV files
│   ├── audit_engines.py            # inspect saved TensorRT engines
│   ├── audit_analysis.py           # audit frozen analysis artifacts
│   ├── bootstrap_ap_ci.py          # paired official-AP bootstrap confidence intervals
│   └── run_bootstrap_ap.sh         # manage a long-running bootstrap job
├── src/
│   ├── dataset.py                  # load COCO, YOLO, and VisDrone annotations
│   ├── calibration.py              # deterministic calibration-set sampling
│   ├── inference.py                # YOLO inference and class validation
│   ├── quantization.py             # TensorRT FP16/INT8 export and inspection
│   ├── evaluation.py               # official COCOeval and saved metrics
│   ├── diagnostics.py              # FP32/INT8 matching and failure diagnostics
│   └── analysis_statistics.py      # bootstrap and adjusted-rate statistics
├── data/
│   ├── metadata/                   # generated dataset statistics
│   └── calibration_lists/          # exact generated calibration IDs
├── outputs/                        # immutable experiment, analysis, and report artifacts
├── pyproject.toml                  # package and dependency definition
├── uv.lock                         # locked dependencies
└── README.md
```

Generated files under `data/` and `outputs/` are intentionally kept outside
version control; preserve them locally because they are the evidence behind
reported metrics.

## 1. Setup

Requirements:

- Linux or WSL2 for TensorRT experiments;
- Python 3.10;
- `uv`;
- NVIDIA driver/CUDA/TensorRT compatible with the `cu130` dependency extra for
  FP16/INT8 experiments.

Keep the virtual environment on the Linux filesystem rather than under
`/mnt/...`, because PyTorch installations contain many small files.

```bash
cd /mnt/d/Project/Quantization

# CPU-only environment: useful for code inspection and non-TensorRT work.
export UV_PROJECT_ENVIRONMENT=/absolute/linux/path/low-bits-small-objects-cpu
uv sync --locked --extra cpu

# CUDA/TensorRT environment: required for FP16-control and real INT8 runs.
# Run this in another shell, or replace the variable above.
export UV_PROJECT_ENVIRONMENT=/absolute/linux/path/low-bits-small-objects-cu130
uv sync --locked --extra cu130
```

The `cpu` and `cu130` extras are mutually exclusive. `pyproject.toml` and
`uv.lock` are the authoritative dependency definitions; `requirements.txt` is
only a compatibility export.

Verify the installation:

```bash
uv run --locked --extra cu130 python scripts/analyze.py --help
uv run --locked --extra cu130 python scripts/run_experiment.py --help
```

## 2. Configure datasets and checkpoints

Edit [config.yaml](config.yaml) and the dataset YAML files before running an
experiment.

```yaml
# config.yaml
paths:
  checkpoint: /absolute/path/to/default-model.pt
  output_root: outputs

runtime:
  device: cuda:0

calibration:
  size: 128
  random_seeds: [0, 1, 2, 3, 4]
```

Each dataset configuration defines its local dataset root and may override the
default checkpoint:

```yaml
model_override:
  checkpoint: /absolute/path/to/dataset-model.pt
  num_classes: 10
```

The checkpoint class count must match the dataset class count. A COCO-80 model,
for example, is not valid for the 10-class VisDrone or 45-class TT100K setup.

Included dataset configurations:

| File | Annotation format | Intended use |
|---|---|---|
| [datasets/coco.yaml](datasets/coco.yaml) | COCO JSON | COCO 2017 experiments |
| [datasets/tt100k.yaml](datasets/tt100k.yaml) | YOLO TXT | TT100K experiments |
| [datasets/visdrone.yaml](datasets/visdrone.yaml) | Native VisDrone TXT | Paper-valid VisDrone evaluation |

The VisDrone configuration uses original annotations, including ignored regions,
so it is suitable for paper-facing evaluation.

The scale thresholds in `config.yaml` must agree with
[protocols/framing.yaml](protocols/framing.yaml) for paper analysis.

## 3. Prepare calibration data

Build training metadata and exact calibration lists once for each dataset:

```bash
uv run --locked --extra cu130 python scripts/prepare_data.py \
  --config config.yaml \
  --dataset datasets/coco.yaml
```

This writes train statistics and the configured random, small-rich, and
scale-balanced lists under `data/`. Each list stores one JSON image ID per line,
preserving numeric and string IDs exactly. If an existing artifact differs, the
command stops rather than replacing it; use `--overwrite` only when replacing
the calibration artifact is intentional and documented.

Repeat this step for `datasets/tt100k.yaml` and the dataset configuration chosen
for VisDrone.

## 4. Run experiments

Start with a small FP32 smoke test:

```bash
uv run --locked --extra cu130 python scripts/run_experiment.py \
  --config config.yaml \
  --dataset datasets/coco.yaml \
  --precision fp32 \
  --max-images 100
```

Then run the full controlled matrix. All runs for a dataset must use the same
checkpoint, validation image IDs, image size, NMS settings, and evaluator.

```bash
# FP32 baseline
uv run --locked --extra cu130 python scripts/run_experiment.py \
  --config config.yaml --dataset datasets/coco.yaml --precision fp32

# Same-backend TensorRT FP16 control
uv run --locked --extra cu130 python scripts/run_experiment.py \
  --config config.yaml --dataset datasets/coco.yaml --precision trt_fp16

# Five independent random INT8 calibration runs
for seed in 0 1 2 3 4; do
  uv run --locked --extra cu130 python scripts/run_experiment.py \
    --config config.yaml --dataset datasets/coco.yaml \
    --precision int8 --calibration random --seed "$seed"
done
```

Optional calibration-ablation runs:

```bash
uv run --locked --extra cu130 python scripts/run_experiment.py \
  --config config.yaml --dataset datasets/coco.yaml \
  --precision int8 --calibration small_rich

uv run --locked --extra cu130 python scripts/run_experiment.py \
  --config config.yaml --dataset datasets/coco.yaml \
  --precision int8 --calibration scale_balanced
```

INT8 means a real TensorRT calibration/export path. If TensorRT is unavailable
or the run fails, the run is marked failed; no simulated result is substituted.

Run directories are named `fp32`, `trt_fp16`, `int8_random_s<seed>`,
`int8_small_rich`, and `int8_scale_balanced`. For a deliberate reproducibility
run, keep prior evidence and use a suffix:

```bash
uv run --locked --extra cu130 python scripts/run_experiment.py \
  --config config.yaml --dataset datasets/coco.yaml \
  --precision fp32 --run-suffix repro_YYYYMMDD
```

Avoid `--overwrite` for paper evidence. It deletes the named run directory and
should only be used for an explicitly disposable run.

## 5. Analyze saved predictions

`analyze.py` never reruns inference. It recomputes official COCO AP from saved
predictions, performs FP32/INT8 matching and diagnostics, and records source
hashes, the resolved config, and the frozen protocol.

```bash
uv run --locked --extra cu130 python scripts/analyze.py \
  --config config.yaml \
  --protocol protocols/framing.yaml \
  --dataset datasets/coco.yaml \
  --dataset datasets/tt100k.yaml \
  --dataset datasets/visdrone.yaml \
  --analysis-id framing_main
```

To include `small_rich` and `scale_balanced` in the analysis:

```bash
uv run --locked --extra cu130 python scripts/analyze.py \
  --config config.yaml \
  --protocol protocols/framing.yaml \
  --dataset datasets/coco.yaml \
  --analysis-id framing_calibration \
  --include-calibration-ablation
```

Results are written to `outputs/analysis/<analysis-id>/`. Key files include:

```text
analysis_manifest.json                 requested inputs and failures
config_resolved.yaml                   frozen analysis configuration
protocol_resolved.yaml                 frozen measurement protocol
official_metrics.csv                   per-run official COCO AP
paper_ap_table.csv                     FP32 vs random-INT8 seed summary
official_degradation_summary.csv       official AP-derived RD, SSR, and SSG
calibration_comparison.csv             optional calibration-ablation summary
figures/*.png and figures/*.csv        figures with exact adjacent source data
<dataset>/<run>/                       matching, bootstrap, regression, FP, diagnostics
```

For an interrupted analysis, use a new `--analysis-id` and pass
`--reuse-analysis <completed-or-partial-analysis-root>`. The protocol must be
byte-identical before completed per-run artifacts are reused.

## 6. Produce report and audits

Build traceable paper tables and figures from a completed analysis:

```bash
uv run --locked --extra cu130 python scripts/report.py \
  --analysis-root outputs/analysis/framing_main \
  --report-id framing_main_report
```

Audit saved FP16 and INT8 TensorRT engines without rebuilding them:

```bash
uv run --locked --extra cu130 python scripts/audit_engines.py \
  --output-root outputs \
  --audit-id framing_main_engines \
  --dataset coco2017 --dataset tt100k --dataset visdrone
```

Audit the frozen analysis for baseline provenance, FP16 control, engine audit
results, failure modes, variance, regression interpretability, and outstanding
bootstrap status:

```bash
uv run --locked --extra cu130 python scripts/audit_analysis.py \
  --analysis-root outputs/analysis/framing_main \
  --resolution-id framing_main_audit \
  --engine-audit-root outputs/engine_audit/framing_main_engines
```

These commands create new immutable output directories under `outputs/`; they
do not alter experiment predictions or rerun inference.

## 7. Bootstrap confidence intervals for official AP

`bootstrap_ap_ci.py` performs image-clustered paired bootstrap confidence
intervals by re-accumulating official COCOeval records. It can take hours for
the full matrix. Use the supervisor script for preflight checks, detached
execution, logging, status, and safe stop confirmation.

```bash
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" \
ANALYSIS_ROOT=outputs/analysis/framing_main \
OUTPUT_ID=framing_main_ap_ci \
DATASET_CONFIGS="datasets/visdrone.yaml datasets/tt100k.yaml datasets/coco.yaml" \
./scripts/run_bootstrap_ap.sh smoke

PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" \
ANALYSIS_ROOT=outputs/analysis/framing_main \
OUTPUT_ID=framing_main_ap_ci \
DATASET_CONFIGS="datasets/visdrone.yaml datasets/tt100k.yaml datasets/coco.yaml" \
./scripts/run_bootstrap_ap.sh start
```

Useful commands:

```bash
./scripts/run_bootstrap_ap.sh status
./scripts/run_bootstrap_ap.sh watch
./scripts/run_bootstrap_ap.sh results
./scripts/run_bootstrap_ap.sh stop
```

The runner refuses to overwrite an existing bootstrap output. Choose a new
`OUTPUT_ID` for a fresh run and preserve partial output from interrupted jobs.

## Artifact policy

Each experiment directory contains at least:

```text
outputs/<dataset>/<run>/
├── config_resolved.yaml
├── run_info.json
├── evaluation_image_ids.json
├── predictions.json
├── rejected_predictions.json          # only when invalid boxes were rejected
├── metrics.json
├── calibration_ids.txt                # INT8 only
└── artifacts/                         # backend settings and saved engine
```

Keep raw `predictions.json` files and calibration-ID lists. They are required
to reproduce official AP, diagnostics, and paper figures. Do not replace them
with aggregate metrics alone.

## Common problems

- **TensorRT unavailable:** use the CUDA/TensorRT environment and verify that
  the installed TensorRT version matches the host driver. Do not substitute a
  fake-quantized result for an INT8 run.
- **Class-count mismatch:** set the checkpoint and `num_classes` in the dataset
  YAML to the matching model/dataset pair.
- **Existing output directory:** select a new run suffix, analysis ID, report
  ID, audit ID, or bootstrap output ID instead of overwriting evidence.
- **VisDrone dataset not found:** update `datasets/visdrone.yaml` to the local
  original image and annotation directories, including ignored regions.
- **Protocol mismatch:** keep `config.yaml` scale bins identical to
  `protocols/framing.yaml` for paper analysis.
