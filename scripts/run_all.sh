#!/usr/bin/env bash
#
# Run every experiment in the paper, then build every figure.
#
#   ./scripts/run_all.sh                      # full paper run (9 seeds x 25 trials)
#   ./scripts/run_all.sh --jobs 100           # more workers
#   ./scripts/run_all.sh --smoke              # tiny config, ~2 min, checks the pipeline
#   ./scripts/run_all.sh default noise        # just these experiments
#   ./scripts/run_all.sh --list               # what can be run
#
# Parallelism lives inside run_experiments.py, not here: it farms out one
# worker per (policy, seed) plus one per seed for the omniscient normalizer,
# and runs the trials sequentially inside each worker. That is already the
# (seed x method) granularity, so this script runs the experiments one after
# another and lets each one saturate the machine. Running experiments
# concurrently on top of that would just oversubscribe the cores and multiply
# the number of concurrent Gurobi solves.
#
# Each experiment is independent: one failing does not stop the rest, and the
# summary at the end says which failed. Per-experiment logs land in
# results/logs/. Partial progress within an experiment survives an
# interruption via results/<experiment>/_progress.jsonl.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONDA_ENV="${CONDA_ENV:-patient}"
JOBS="${JOBS:-63}"
NUM_SEEDS="${NUM_SEEDS:-9}"
NUM_TRIALS="${NUM_TRIALS:-25}"
OUT_DIR="${OUT_DIR:-$REPO/results}"
EXTRA_ARGS=()
SKIP_FIGURES=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo "Options:"
    echo "  --jobs N          worker processes per experiment (default $JOBS)"
    echo "  --num-seeds N     independent environment draws (default $NUM_SEEDS)"
    echo "  --num-trials N    theta/order draws per seed (default $NUM_TRIALS)"
    echo "  --out-dir DIR     results directory (default results/)"
    echo "  --smoke           tiny config to check the pipeline end to end"
    echo "  --skip-figures    run experiments only, do not build figures"
    echo "  --list            list experiment names and exit"
    echo "  --                everything after this is passed to run_experiments.py"
}

# gurobipy lives in the conda env, so activate it unless we are already there.
# `set +u` around this on purpose: conda's own activate.d hooks (geotiff's,
# for one) read unset variables and would abort the script under `set -u`.
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
fi
python -c "import gurobipy" 2>/dev/null || {
    echo "ERROR: gurobipy not importable in env '$CONDA_ENV'." >&2; exit 1; }

export PYTHONPATH="$REPO"

# The canonical experiment list lives in run_experiments.py; read it rather
# than duplicating it here. Must come after the conda activation above --
# importing run_experiments needs gurobipy. Empty output means something is
# broken, so fail loudly instead of silently running nothing.
mapfile -t ALL_EXPERIMENTS < <(python -c \
    "from scripts.run_experiments import EXPERIMENTS; print('\n'.join(EXPERIMENTS))")
if [[ ${#ALL_EXPERIMENTS[@]} -eq 0 ]]; then
    echo "ERROR: could not read the experiment list from scripts/run_experiments.py" >&2
    exit 1
fi

EXPERIMENTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)        JOBS="$2"; shift 2 ;;
        --num-seeds)   NUM_SEEDS="$2"; shift 2 ;;
        --num-trials)  NUM_TRIALS="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2"; shift 2 ;;
        --skip-figures) SKIP_FIGURES=1; shift ;;
        --smoke)       NUM_SEEDS=2; NUM_TRIALS=2; JOBS=16
                       EXTRA_ARGS+=(--N 60 --M 40 --small-N 8 --small-M 6 --small-k 2)
                       shift ;;
        --list)        printf '%s\n' "${ALL_EXPERIMENTS[@]}"; exit 0 ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; EXTRA_ARGS+=("$@"); break ;;
        -*)            echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)             EXPERIMENTS+=("$1"); shift ;;
    esac
done
[[ ${#EXPERIMENTS[@]} -eq 0 ]] && EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=================================================================="
echo " repo        $REPO"
echo " env         $CONDA_ENV"
echo " experiments ${EXPERIMENTS[*]}"
echo " seeds       $NUM_SEEDS   trials $NUM_TRIALS   jobs $JOBS"
echo " out-dir     $OUT_DIR"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo " extra       ${EXTRA_ARGS[*]}"
echo " logs        $LOG_DIR/<experiment>_$STAMP.log"
echo "=================================================================="

declare -a STATUS_LINES=()
OVERALL=0
RUN_START=$SECONDS

for exp in "${EXPERIMENTS[@]}"; do
    log="$LOG_DIR/${exp}_${STAMP}.log"
    echo ""
    echo ">>> $exp   (log: $log)"
    start=$SECONDS
    python -W ignore scripts/run_experiments.py \
        --experiment "$exp" \
        --num-seeds "$NUM_SEEDS" \
        --num-trials "$NUM_TRIALS" \
        --jobs "$JOBS" \
        --out-dir "$OUT_DIR" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$log" 2>&1
    rc=$?
    elapsed=$((SECONDS - start))
    if [[ $rc -eq 0 ]]; then
        # surface each config's headline number without dumping the whole log
        grep -h "norm_util=" "$log" | sed 's/^/    /' || true
        STATUS_LINES+=("$(printf '  %-24s ok      %5ds' "$exp" "$elapsed")")
        echo "    done in ${elapsed}s"
    else
        OVERALL=1
        STATUS_LINES+=("$(printf '  %-24s FAILED  %5ds  (rc=%d, see %s)' "$exp" "$elapsed" "$rc" "$log")")
        echo "    FAILED (rc=$rc) -- continuing; see $log"
        tail -n 15 "$log" | sed 's/^/    | /'
    fi
done

if [[ $SKIP_FIGURES -eq 0 ]]; then
    log="$LOG_DIR/figures_${STAMP}.log"
    echo ""
    echo ">>> figures   (log: $log)"
    start=$SECONDS
    python -W ignore scripts/make_figures.py --figure all \
        --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures" > "$log" 2>&1
    rc=$?
    grep -h -E "^saved|^skip" "$log" | sed 's/^/    /' || true
    if [[ $rc -eq 0 ]]; then
        STATUS_LINES+=("$(printf '  %-24s ok      %5ds' "figures" "$((SECONDS - start))")")
    else
        OVERALL=1
        STATUS_LINES+=("$(printf '  %-24s FAILED  %5ds  (see %s)' "figures" "$((SECONDS - start))" "$log")")
        tail -n 15 "$log" | sed 's/^/    | /'
    fi
fi

echo ""
echo "=================================================================="
echo " summary   (total $((SECONDS - RUN_START))s)"
printf '%s\n' "${STATUS_LINES[@]}"
echo "=================================================================="
exit $OVERALL
