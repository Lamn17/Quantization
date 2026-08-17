#!/usr/bin/env bash
# Supervise the offline scale-wise AP bootstrap.
#
# The underlying job runs for hours, so it is detached from the calling terminal
# and every command here is safe to run against a job that is already going.
#
#   ./scripts/run_bootstrap_ap.sh smoke     20 replicates on one dataset, minutes
#   ./scripts/run_bootstrap_ap.sh start     the real 1000-replicate run
#   ./scripts/run_bootstrap_ap.sh status    progress, rate, projection
#   ./scripts/run_bootstrap_ap.sh watch     follow the log
#   ./scripts/run_bootstrap_ap.sh results   print the finished intervals
#   ./scripts/run_bootstrap_ap.sh stop      terminate the job
#
# Override any default from the environment, for example
#   REPLICATES=200 OUTPUT_ID=quick ./scripts/run_bootstrap_ap.sh start

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-.venv/bin/python}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-outputs/analysis/framing_main}
OUTPUT_ID=${OUTPUT_ID:-partab_20260814}
REPLICATES=${REPLICATES:-1000}
# Smallest split first, so the dataset the variance question is about lands early.
DATASET_CONFIGS=${DATASET_CONFIGS:-"datasets/visdrone.yaml datasets/tt100k.yaml datasets/coco.yaml"}
RUNS_PER_DATASET=6 # one FP32 baseline plus five INT8 calibration seeds

LOG_DIR=logs
LOG="$LOG_DIR/bootstrap_ap_${OUTPUT_ID}.log"
RESULT_DIR="outputs/bootstrap_ap/${OUTPUT_ID}"

if [[ -t 1 ]]; then
    red() { printf '\033[31m%s\033[0m\n' "$*"; }
    green() { printf '\033[32m%s\033[0m\n' "$*"; }
    bold() { printf '\033[1m%s\033[0m\n' "$*"; }
else
    red() { printf '%s\n' "$*"; }
    green() { printf '%s\n' "$*"; }
    bold() { printf '%s\n' "$*"; }
fi

# Match on the output id so several ids can run side by side without confusion.
job_pid() {
    pgrep -f "bootstrap_ap_ci\.py.*--output-id[= ]${OUTPUT_ID}( |$)" 2>/dev/null | head -1
}

# A job started by hand writes wherever its shell redirected, so resolve the
# real stdout target rather than assuming this script's naming scheme.
resolve_log() {
    local pid=$1
    [[ -n "$pid" ]] || return 1
    local target
    target=$(readlink "/proc/${pid}/fd/1" 2>/dev/null) || return 1
    [[ -f "$target" ]] || return 1
    printf '%s\n' "$target"
}

require_not_running() {
    local pid
    pid=$(job_pid || true)
    if [[ -n "$pid" ]]; then
        red "A job for output-id '${OUTPUT_ID}' is already running (pid ${pid})."
        echo "Use 'status' to inspect it, or 'stop' to terminate it first."
        exit 1
    fi
}

preflight() {
    bold "Preflight"
    [[ -x "$PYTHON" ]] || {
        red "Interpreter not executable: $PYTHON"
        exit 1
    }
    if [[ -e "$RESULT_DIR" ]]; then
        red "Output directory already exists: $RESULT_DIR"
        echo "bootstrap_ap_ci.py refuses to overwrite results. Either:"
        echo "  rm -rf $RESULT_DIR"
        echo "  OUTPUT_ID=<other-name> $0 ${1:-start}"
        exit 1
    fi
    # Validate every input the job will touch, so a missing prediction file
    # surfaces now instead of after the first dataset has already been paid for.
    "$PYTHON" - "$ANALYSIS_ROOT" $DATASET_CONFIGS <<'PY'
import sys
from pathlib import Path

import yaml

analysis_root = Path(sys.argv[1])
dataset_configs = [Path(item) for item in sys.argv[2:]]
problems: list[str] = []

for required in ("protocol_resolved.yaml", "config_resolved.yaml"):
    if not (analysis_root / required).is_file():
        problems.append(f"missing {analysis_root / required}")

config_path = analysis_root / "config_resolved.yaml"
if config_path.is_file():
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    seeds = [int(value) for value in config["calibration"]["random_seeds"]]
    runs = ["fp32"] + [f"int8_random_s{seed}" for seed in seeds]
    for dataset_config in dataset_configs:
        if not dataset_config.is_file():
            problems.append(f"missing dataset config {dataset_config}")
            continue
        document = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
        dataset = document["dataset"] if "dataset" in document else document
        name = dataset["name"]
        root = Path(str(dataset["root"]).strip())
        if not root.is_dir():
            problems.append(f"{name}: dataset root not found: {root}")
        for run in runs:
            predictions = output_root / name / run / "predictions.json"
            if not predictions.is_file():
                problems.append(f"{name}: missing {predictions}")
        print(f"  {name:10s} {len(runs)} runs, root {root}")

if problems:
    print("\nPreflight failed:")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)
print("  all inputs present")
PY
    green "Preflight passed"
}

launch() {
    local replicates=$1
    shift
    local configs=("$@")
    mkdir -p "$LOG_DIR"
    # Create the log before the job starts. On /mnt drvfs mounts a freshly
    # redirected file can be briefly invisible, which is what makes a
    # immediately-following 'tail -f' fail with "No such file or directory".
    : >"$LOG"
    local arguments=(
        "$PYTHON" scripts/bootstrap_ap_ci.py
        --analysis-root "$ANALYSIS_ROOT"
        --output-id "$OUTPUT_ID"
        --replicates "$replicates"
    )
    local config
    for config in "${configs[@]}"; do
        arguments+=(--dataset-config "$config")
    done
    # setsid detaches the job into its own session, so closing the terminal
    # cannot reach it; nohup alone only blocks SIGHUP.
    if command -v setsid >/dev/null 2>&1; then
        setsid nohup "${arguments[@]}" >>"$LOG" 2>&1 &
    else
        nohup "${arguments[@]}" >>"$LOG" 2>&1 &
    fi
    sleep 2
    local pid
    pid=$(job_pid || true)
    if [[ -z "$pid" ]]; then
        red "Job exited immediately. Log follows:"
        cat "$LOG"
        exit 1
    fi
    green "Started pid ${pid}, ${replicates} replicates, output-id ${OUTPUT_ID}"
    echo "  log     $LOG"
    echo "  results $RESULT_DIR"
    echo
    echo "Follow it with:  $0 watch"
    echo "Check it with:   $0 status"
    echo "The job is detached; closing this terminal will not stop it."
}

command_start() {
    require_not_running
    # shellcheck disable=SC2086 # word splitting is the intended list form
    preflight start
    bold "Starting"
    # shellcheck disable=SC2086
    launch "$REPLICATES" $DATASET_CONFIGS
}

command_smoke() {
    OUTPUT_ID="${OUTPUT_ID}_smoke"
    LOG="$LOG_DIR/bootstrap_ap_${OUTPUT_ID}.log"
    RESULT_DIR="outputs/bootstrap_ap/${OUTPUT_ID}"
    require_not_running
    local first
    first=$(echo $DATASET_CONFIGS | awk '{print $1}')
    preflight smoke
    bold "Starting smoke run"
    echo "  one dataset ($first), 20 replicates, discard when it passes"
    launch 20 "$first"
}

command_status() {
    local pid
    pid=$(job_pid || true)
    # Prefer the log the process is actually writing to.
    local discovered
    discovered=$(resolve_log "$pid" || true)
    if [[ -n "$discovered" && "$discovered" != "$(readlink -f "$LOG" 2>/dev/null)" ]]; then
        LOG="$discovered"
        echo "Log discovered from the running process: $LOG"
    fi
    if [[ -n "$pid" ]]; then
        local elapsed rss
        elapsed=$(ps -o etime= -p "$pid" | tr -d ' ')
        rss=$(ps -o rss= -p "$pid" | tr -d ' ')
        green "RUNNING  pid ${pid}  elapsed ${elapsed}  rss $((rss / 1024)) MiB"
    elif [[ -f "$LOG" ]] && grep -q "bootstrap_ap_root" "$LOG"; then
        green "FINISHED"
    elif [[ -f "$LOG" ]]; then
        red "NOT RUNNING and no completion marker in the log"
        echo "Last lines:"
        tail -5 "$LOG" | sed 's/^/  /'
    else
        echo "No job and no log for output-id '${OUTPUT_ID}'."
        return 0
    fi

    local total_runs datasets_count runs_done
    datasets_count=$(echo $DATASET_CONFIGS | wc -w)
    total_runs=$((datasets_count * RUNS_PER_DATASET))
    runs_done=$(grep -c ': done in ' "$LOG" 2>/dev/null || true)
    runs_done=${runs_done:-0}
    echo "  runs finished        ${runs_done}/${total_runs}"
    local datasets_done
    datasets_done=$(grep -c ': intervals written' "$LOG" 2>/dev/null || true)
    echo "  datasets finished    ${datasets_done:-0}/${datasets_count}"
    local current
    current=$(grep -E ': (evaluate\(\)|[0-9.]+s per replicate)' "$LOG" 2>/dev/null | tail -1 || true)
    [[ -n "$current" ]] && echo "  latest log line      ${current}"
    local rate
    rate=$(grep 'per replicate' "$LOG" 2>/dev/null | tail -1 || true)
    if [[ -n "$rate" ]]; then
        echo "  latest projection    $(echo "$rate" | sed 's/.*projected //')"
    fi
    if [[ -d "$RESULT_DIR" ]]; then
        echo "  interval files       $(find "$RESULT_DIR" -name 'intervals.json' | wc -l)"
    fi
}

command_watch() {
    local discovered
    discovered=$(resolve_log "$(job_pid || true)" || true)
    [[ -n "$discovered" ]] && LOG="$discovered"
    [[ -f "$LOG" ]] || {
        red "No log yet: $LOG"
        exit 1
    }
    echo "Following $LOG  (Ctrl-C leaves the job running)"
    tail -f "$LOG"
}

command_stop() {
    local pid
    pid=$(job_pid || true)
    [[ -n "$pid" ]] || {
        echo "Nothing running for output-id '${OUTPUT_ID}'."
        exit 0
    }
    # Killing a multi-hour job discards every unwritten dataset, so require an
    # explicit confirmation. Set FORCE=1 to skip it in a script.
    if [[ "${FORCE:-0}" != "1" ]]; then
        local elapsed
        elapsed=$(ps -o etime= -p "$pid" | tr -d ' ')
        red "About to terminate pid ${pid} (output-id ${OUTPUT_ID}, running ${elapsed})."
        echo "Unwritten datasets will be lost and cannot be resumed."
        if [[ ! -t 0 ]]; then
            red "Refusing to stop without a terminal. Re-run with FORCE=1 to confirm."
            exit 1
        fi
        read -r -p "Type the output-id to confirm: " reply
        if [[ "$reply" != "$OUTPUT_ID" ]]; then
            echo "Mismatch, leaving the job alone."
            exit 1
        fi
    fi
    red "Terminating pid ${pid}"
    kill "$pid"
    sleep 3
    if [[ -n "$(job_pid || true)" ]]; then
        red "Still alive, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    echo "Stopped. Partial results, if any, remain in $RESULT_DIR"
    echo "Remove that directory before starting again."
}

command_results() {
    [[ -d "$RESULT_DIR" ]] || {
        red "No results yet: $RESULT_DIR"
        exit 1
    }
    "$PYTHON" - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("*/intervals.json"))
if not files:
    print(f"No intervals.json under {root} yet.")
    raise SystemExit(0)

print("Paired delta-AP versus FP32, averaged over calibration seeds")
print("This is the Table 1 row set. Image-clustered bootstrap, 95 percent.\n")
header = f"{'dataset':10s} {'metric':6s} {'median':>9s} {'lower':>9s} {'upper':>9s} {'width':>8s}"
print(header)
print("-" * len(header))
for path in files:
    for row in json.load(path.open()):
        if row["run"] != "int8_random_seed_mean":
            continue
        if row["lower_95"] is None:
            continue
        width = row["upper_95"] - row["lower_95"]
        print(
            f"{row['dataset']:10s} {row['metric']:6s} {row['median']:9.4f} "
            f"{row['lower_95']:9.4f} {row['upper_95']:9.4f} {width:8.4f}"
        )
print(
    "\nThese intervals cover evaluation-set resampling within a calibration "
    "seed only.\nFor VisDrone small and medium the spread across seeds "
    "dominates by 54x and 30x,\nso report those cells with the between-seed "
    "deviation as well; see\noutputs/issue_resolution/*/c2_variance_decomposition.csv"
)
PY
}

case "${1:-help}" in
start) command_start ;;
smoke) command_smoke ;;
status) command_status ;;
watch) command_watch ;;
stop) command_stop ;;
results) command_results ;;
help | -h | --help)
    # Print the leading comment block only, stopping at the first line of code.
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
        "${BASH_SOURCE[0]}"
    ;;
*)
    red "Unknown command: $1"
    echo "Try: $0 help"
    exit 1
    ;;
esac
