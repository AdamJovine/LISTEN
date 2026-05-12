#!/usr/bin/env bash
# Generate every plot that the existing plotting scripts can produce from
# the runs in the resume_paper output directory.
#
# Current sweep is tournament-only: section-order @ B=32 across 9 pairs and
# a batch-size sweep over the 4 canonical pairs. Skipped scripts:
#   - headphones_plot.py: needs LISTEN-U (utility) data — not in this sweep.
#   - plot_order_study.py: compares header_then_task_v1 vs task_then_header_v1
#     but the new sweep only has the former.
#
# Usage:
#   bash scripts/make_plots.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/adamjovine/Documents/IJCAI/LISTEN-IJCAI/outputs/paper__REPS40__iters25__seed1234__20260511_163727}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"
mkdir -p "${PLOT_DIR}"

MAIN_BATCH_SIZE=32

echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"
echo "[PLOT_DIR]    ${PLOT_DIR}"
echo

# Plot 1: Per-scenario × batch sizes (tournament)
echo "═══ [PLOT 1] Per-scenario × batch sizes (tournament) ═══"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/general_plot.py" \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large algo tournament scenario all mode all api_model all \
  --x_medium batch_size all \
  --y nar \
  || echo "[WARN] PLOT 2 failed"

# Plot 2: Cross-scenario aggregate (tournament @ B=MAIN_BATCH_SIZE) per api
echo "═══ [PLOT 2] Cross-scenario × tournament @ B=${MAIN_BATCH_SIZE} ═══"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/general_plot.py" \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large api_model all \
  --x_medium scenario all \
  --x_small algo all \
  --y nar \
  --canonical_mode \
  --per_algo_filter tournament batch_size "${MAIN_BATCH_SIZE}" \
  || echo "[WARN] PLOT 2 failed"

# Plot 3: BASE vs canonical mode (tournament traces only; utility absent)
# Pinned to B=32 + default section order so each bar averages exactly 40 reps
# of one configuration, isolating the BASE-vs-canonical effect.
echo "═══ [PLOT 3] BASE vs canonical mode @ B=${MAIN_BATCH_SIZE}, default order ═══"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_base_study.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  --section-order "persona,attributes,priorities" \
  || echo "[WARN] PLOT 3 failed"

echo
echo "[DONE] Plots written to ${PLOT_DIR}"
