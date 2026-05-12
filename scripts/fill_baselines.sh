#!/usr/bin/env bash
# Ensure 40 baseline runs exist for headphones:MAIN and exam:REGISTRAR @ groq,
# then regenerate groq__nar__scenario__by_algo.png so the BaselineRandom +
# BaselineZscore series populate. Idempotent: if data is already full, the
# baseline loop is a no-op and only the plot is refreshed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

unset GEMINI_API_KEY GOOGLE_API_KEY GROQ_API_KEY

OUTPUT_ROOT="${OUTPUT_ROOT:-/Users/adamjovine/Documents/IJCAI/LISTEN-IJCAI/outputs/paper__REPS40__iters25__seed1234__20260511_163727}"
PLOT_DIR="${OUTPUT_ROOT}/plots"
TARGET_REPS="${TARGET_REPS:-40}"
BASE_SEED="${BASE_SEED:-90000}"
GROUP_STAMP="$(date -u +'%Y%m%d_%H%M%S')"
mkdir -p "${PLOT_DIR}"

PAIRS=(
  "headphones:MAIN"
  "exam:REGISTRAR"
)

count_baseline_groq() {
  local scen="$1" mode="$2"
  SCEN_F="${scen}" MODE_F="${mode}" OUTPUT_ROOT_F="${OUTPUT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import json, os, sys
from pathlib import Path
root = Path(os.environ["OUTPUT_ROOT_F"]) / os.environ["SCEN_F"]
scen, mode = os.environ["SCEN_F"], os.environ["MODE_F"]
if not root.is_dir(): print(0); sys.exit()
n = 0
for p in root.glob("*.json"):
    try: m = json.loads(p.read_text()).get("meta", {})
    except Exception: continue
    if (m.get("algo")=="baseline" and m.get("scenario")==scen
        and m.get("mode")==mode and m.get("api_model")=="groq"):
        n += 1
print(n)
PY
}

for pair in "${PAIRS[@]}"; do
  IFS=":" read -r scen mode <<<"${pair}"
  existing=$(count_baseline_groq "${scen}" "${mode}")
  remaining=$((TARGET_REPS - existing))
  echo "[baseline ${scen}/${mode}/groq] ${existing}/${TARGET_REPS} — running ${remaining}"
  for ((i=0; i<remaining; i++)); do
    seed=$((BASE_SEED + i))
    "${PYTHON_BIN}" run_algorithm.py \
      --algo baseline --scenario "${scen}" --mode "${mode}" \
      --api-model groq --iterations 25 --seed "${seed}" \
      --output-root "${OUTPUT_ROOT}/${scen}" --run-stamp "${GROUP_STAMP}" \
      || echo "[WARN] baseline ${scen}/${mode}/seed${seed} failed"
  done
done

echo
echo "[plot] regenerating groq__nar__scenario__by_algo.png"
"${PYTHON_BIN}" plotting/general_plot.py \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large api_model groq \
  --x_medium scenario all \
  --x_small algo all \
  --y nar \
  --canonical_mode \
  --per_algo_filter tournament batch_size 32

echo "[DONE] ${PLOT_DIR}/groq__nar__scenario__by_algo.png"
