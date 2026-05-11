#!/usr/bin/env bash
# Sweep concordance (paper §4.1) over every (scenario, mode) pair with a
# human_sol entry in the repo configs.
#
# Usage:
#   bash post_analysis/run_concordance_analysis.sh
#
# Optional env overrides:
#   N_SAMPLES=10000  RANDOM_SEED=42  PYTHON_BIN=python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

N_SAMPLES="${N_SAMPLES:-10000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
STAMP="$(date -u +"%Y%m%d_%H%M%S")"
OUTPUT_CSV="${SCRIPT_DIR}/concordance_results_n${N_SAMPLES}_${STAMP}.csv"

PAIRS=(
  "flight00:Complicated_structured"
  "flight02:Complicated"
  "exam:REGISTRAR"
  "headphones:STUDENT"
  "headphones:STUDENT_HARD"
)

echo "[concordance] N_SAMPLES=${N_SAMPLES} RANDOM_SEED=${RANDOM_SEED}"
echo "[concordance] writing -> ${OUTPUT_CSV}"
echo "scenario,mode,concordance,concordance_se,concordance_lower,concordance_upper,n_samples" > "${OUTPUT_CSV}"

for pair in "${PAIRS[@]}"; do
  scen="${pair%%:*}"
  mode="${pair##*:}"
  echo ""
  echo "── ${scen} / ${mode} ──────────────────────────────────"
  ROW_CSV="$(mktemp -t concordance_row_XXXXXX.csv)"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/concordance_analysis.py" \
    --scenario "${scen}" \
    --mode "${mode}" \
    --n-samples "${N_SAMPLES}" \
    --random-seed "${RANDOM_SEED}" \
    --output "${ROW_CSV}"
  tail -n +2 "${ROW_CSV}" >> "${OUTPUT_CSV}"
  rm -f "${ROW_CSV}"
done

echo ""
echo "[concordance] done -> ${OUTPUT_CSV}"
