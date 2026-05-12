#!/usr/bin/env bash
# Resume the section-order + batch-size sweep into an existing run directory.
# paper_recreate.sh already uses count_runs (meta-based dedup) to skip any
# (algo, scenario, mode, api, batch_size, prompt_variant, section_order) tuple
# that has already reached TARGET_REPS — this wrapper just pins OUTPUT_ROOT
# at an existing dir and clears any stale API keys from the shell so .env wins.
#
# Usage:
#   bash scripts/resume_paper.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset GEMINI_API_KEY GOOGLE_API_KEY GROQ_API_KEY

OUTPUT_ROOT="/Users/adamjovine/Documents/IJCAI/LISTEN-IJCAI/outputs/paper__REPS40__iters25__seed1234__20260511_163727"

echo "[dedupe] cleaning duplicate baseline zscore_winner entries before resume"
python "${SCRIPT_DIR}/dedupe_zscore.py" "${OUTPUT_ROOT}" || \
  echo "[WARN] dedupe_zscore.py failed (continuing)"

OUTPUT_ROOT="${OUTPUT_ROOT}" exec bash "${SCRIPT_DIR}/paper_recreate.sh"
