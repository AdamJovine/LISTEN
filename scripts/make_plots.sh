#!/usr/bin/env bash
# Generate the order-sensitivity plots from a run directory.
#
# Usage:
#   OUTPUT_ROOT=outputs/<run-dir> bash scripts/make_plots.sh
#
# If OUTPUT_ROOT is omitted, the latest matching directory under outputs/ is
# used. API_MODEL defaults to groq; set API_MODEL=gemini or
# API_MODEL=groq,gemini after the other model's runs are complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ ! -d "${REPO_ROOT}/outputs" ]]; then
    echo "[ERR] Set OUTPUT_ROOT=<run directory>; ${REPO_ROOT}/outputs does not exist." >&2
    exit 1
  fi
  shopt -s nullglob
  candidates=(
    "${REPO_ROOT}/outputs"/order_sensitivity__*
    "${REPO_ROOT}/outputs"/paper__REPS*
  )
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "[ERR] Set OUTPUT_ROOT=<run directory>; no matching outputs were found." >&2
    exit 1
  fi
  latest_output="$(ls -td -- "${candidates[@]}" | head -n 1)"
  OUTPUT_ROOT="${latest_output}"
fi

PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/outputs/plots}"
mkdir -p "${PLOT_DIR}"

MAIN_BATCH_SIZE="${MAIN_BATCH_SIZE:-32}"
DEFAULT_ORDER="${DEFAULT_ORDER:-persona,priorities,attributes}"
API_MODEL="${API_MODEL:-groq}"

api_args=()
IFS="," read -ra api_models <<<"${API_MODEL}"
for api in "${api_models[@]}"; do
  api="${api//[[:space:]]/}"
  [[ -n "${api}" ]] && api_args+=(--api-model "${api}")
done

general_api_value="${API_MODEL}"
if [[ "${API_MODEL}" == *","* ]]; then
  general_api_value="all"
fi

echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"
echo "[PLOT_DIR]    ${PLOT_DIR}"
echo

echo "=== [PLOT 1] Per-scenario by batch size (tournament) ==="
# --canonical_mode restricts each scenario to its canonical (with-preference)
# mode; without it, Section 1 leaks B=32 BASE/SOFT data and produces one-point
# "sweeps" for those modes.
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/general_plot.py" \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large algo tournament scenario all api_model "${general_api_value}" \
  --x_medium batch_size all \
  --canonical_mode \
  --y nar \
  || echo "[WARN] PLOT 1 failed"

echo "=== [PLOT 2] Cross-scenario by algo @ B=${MAIN_BATCH_SIZE}, default order ==="
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_orders_by_algo.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  --section-order "${DEFAULT_ORDER}" \
  --aggregate-orders \
  "${api_args[@]}" \
  || echo "[WARN] PLOT 2 failed"

echo "=== [PLOT 2b] Cross-scenario by algo and section order @ B=${MAIN_BATCH_SIZE} ==="
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_orders_by_algo.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  "${api_args[@]}" \
  || echo "[WARN] PLOT 2b failed"

echo "=== [PLOT 3] BASE vs canonical mode @ B=${MAIN_BATCH_SIZE}, default order ==="
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_base_study.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --batch-size "${MAIN_BATCH_SIZE}" \
  --section-order "${DEFAULT_ORDER}" \
  "${api_args[@]}" \
  || echo "[WARN] PLOT 3 failed"

echo
echo "[DONE] Plots written to ${PLOT_DIR}"
