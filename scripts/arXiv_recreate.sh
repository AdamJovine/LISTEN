#!/usr/bin/env bash
# Full arXiv pipeline: run the paper sweep (paper_recreate.sh) and produce
# every canonical plot under outputs/plots/.
#
# Stage 1 — experiments. Delegates to scripts/paper_recreate.sh:
#   Section 0: utility (6 orders) + baseline + full_batch, 9 pairs × 2 apis.
#   Section 1: tournament section-order sweep @ B=32, 6 orders × 9 pairs × 2 apis.
#   Section 2: tournament batch-size sweep, 4 canonical pairs × {2,4,8,16,32} × 2 apis.
#
# Stage 2 — plots into ${REPO_ROOT}/outputs/plots/:
#   - general_plot:        TournamentExperiment__<scenario>__<mode>__<api>__nar__batch_size.png
#   - plot_orders_by_algo: <api>__nar__scenario__by_algo.{png,csv} (aggregated)
#   - plot_orders_by_algo: <api>__nar__scenario__by_algo_orders.{png,csv} (per section_order)
#   - plot_base_study:     base_study_nar__<api>.png + base_vs_primary_table__<api>.csv
#   - headphones_plot:     headphones__MODE__<api>__batch8__norm-avg-rank-both.png
#
# Usage:
#   bash scripts/arXiv_recreate.sh
#
# Optional env overrides (forwarded to paper_recreate.sh where relevant):
#   TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 PYTHON_BIN=python
#   OUTPUT_ROOT=<path>     # resume into an existing run dir
#   PLOT_DIR=<path>        # override plot destination (default outputs/plots)
#   SKIP_RUNS=1            # only do Stage 2 (assumes OUTPUT_ROOT already populated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

# ── Resolve OUTPUT_ROOT before delegating so Stage 2 can find Stage 1's output.
TARGET_REPS="${TARGET_REPS:-40}"
ITERS="${ITERS:-25}"
BASE_SEED="${BASE_SEED:-1234}"
GROUP_STAMP="${GROUP_STAMP:-$(date -u +"%Y%m%d_%H%M%S")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper__REPS${TARGET_REPS}__iters${ITERS}__seed${BASE_SEED}__${GROUP_STAMP}}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/outputs/plots}"

export PYTHON_BIN TARGET_REPS ITERS BASE_SEED GROUP_STAMP OUTPUT_ROOT
mkdir -p "${PLOT_DIR}"

echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"
echo "[PLOT_DIR]    ${PLOT_DIR}"
echo

# Keep legacy scenario aliases in plot filenames for compatibility:
#   flights_chi_nyc       -> flight00
#   flights_ithaca_reston -> flight02
copy_legacy_batch_plot_aliases() {
  local copied=0
  local src dst

  shopt -s nullglob

  for src in "${PLOT_DIR}"/TournamentExperiment__flights_chi_nyc__*__nar__batch_size.png; do
    dst="${src/__flights_chi_nyc__/__flight00__}"
    cp -f "${src}" "${dst}"
    copied=$((copied + 1))
  done

  for src in "${PLOT_DIR}"/TournamentExperiment__flights_ithaca_reston__*__nar__batch_size.png; do
    dst="${src/__flights_ithaca_reston__/__flight02__}"
    cp -f "${src}" "${dst}"
    copied=$((copied + 1))
  done

  shopt -u nullglob
  echo "─── Legacy aliases written: ${copied} batch-size plot(s)"
}

# ── Stage 1: experiments ───────────────────────────────────────────────────
if [[ "${SKIP_RUNS:-0}" != "1" ]]; then
  echo "═══════════════════════════════════════════════════════════════════"
  echo "[STAGE 1] Running paper sweep via paper_recreate.sh"
  echo "═══════════════════════════════════════════════════════════════════"
  bash "${SCRIPT_DIR}/paper_recreate.sh"
else
  echo "[STAGE 1] SKIP_RUNS=1 — skipping paper_recreate.sh"
fi

# ── Stage 2: plots ─────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════"
echo "[STAGE 2] Generating plots into ${PLOT_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

MAIN_BATCH_SIZE="${MAIN_BATCH_SIZE:-32}"
DEFAULT_ORDER="${DEFAULT_ORDER:-persona,attributes,priorities}"
HEADPHONES_BATCH_SIZE="${HEADPHONES_BATCH_SIZE:-8}"
API_MODELS=("groq" "gemini")

echo "─── [PLOT 1] Per-scenario × batch sizes (tournament) ───"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/general_plot.py" \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large algo tournament scenario all mode all api_model all \
  --x_medium batch_size all \
  --y nar \
  || echo "[WARN] general_plot.py failed"
copy_legacy_batch_plot_aliases

echo "─── [PLOT 2] Cross-scenario × algo @ B=${MAIN_BATCH_SIZE}, default order (aggregated) ───"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_orders_by_algo.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  --section-order "${DEFAULT_ORDER}" \
  --aggregate-orders \
  || echo "[WARN] plot_orders_by_algo (aggregated) failed"

echo "─── [PLOT 2b] Cross-scenario × algo × section_order @ B=${MAIN_BATCH_SIZE} ───"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_orders_by_algo.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  || echo "[WARN] plot_orders_by_algo (by-order) failed"

echo "─── [PLOT 3] BASE vs canonical mode @ B=${MAIN_BATCH_SIZE}, default order ───"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_base_study.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  --section-order "${DEFAULT_ORDER}" \
  || echo "[WARN] plot_base_study.py failed"

echo "─── [PLOT 4] Headphones LISTEN-T vs LISTEN-U @ B=${HEADPHONES_BATCH_SIZE} ───"
for api in "${API_MODELS[@]}"; do
  "${PYTHON_BIN}" "${REPO_ROOT}/plotting/headphones_plot.py" \
    --scenario headphones \
    --batch-size "${HEADPHONES_BATCH_SIZE}" \
    --api-model "${api}" \
    --output-dir "${OUTPUT_ROOT}/headphones" \
    --plot-dir "${PLOT_DIR}" \
    || echo "[WARN] headphones_plot.py (${api}) failed"
done

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "[DONE] Runs:  ${OUTPUT_ROOT}"
echo "[DONE] Plots: ${PLOT_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
