#!/usr/bin/env bash
# Run the section-order prompt sensitivity sweep and generate its plots.
#
# This complements scripts/paper_recreate.sh. It does not replace the paper
# reproduction workflow; it runs the PR #9 order-sensitivity grid needed for
# plots such as groq__nar__scenario__by_algo_orders.png.
#
# Usage:
#   bash scripts/order_sensitivity_recreate.sh
#
# Optional env overrides:
#   TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 PYTHON_BIN=python
#   API_MODELS=groq          # comma/space-separated; default is groq
#   OUTPUT_ROOT=<path>     # resume into an existing run dir
#   RUN_PLOTS=0            # skip plot generation at the end

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ALGO="${REPO_ROOT}/run_algorithm.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

TARGET_REPS="${TARGET_REPS:-40}"
BASELINE_REPS="${BASELINE_REPS:-${TARGET_REPS}}"
ITERS="${ITERS:-25}"
BASE_SEED="${BASE_SEED:-1234}"
JOBS="${JOBS:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
RUN_PLOTS="${RUN_PLOTS:-1}"

DEFAULT_PROMPT="header_then_task_v1"
MAIN_BATCH_SIZE=32
SWEEP_BATCH_SIZES=(2 4 8 16 32)

API_MODEL_LIST="${API_MODELS:-groq}"
API_MODEL_LIST="${API_MODEL_LIST//,/ }"
read -r -a API_MODELS <<<"${API_MODEL_LIST}"
if [[ ${#API_MODELS[@]} -eq 0 ]]; then
  echo "[ERR] API_MODELS resolved to an empty list." >&2
  exit 1
fi

ALL_PAIRS=(
  "flights_ithaca_reston:Complicated"
  "flights_chi_nyc:Complicated_structured"
  "exam:REGISTRAR"
  "headphones:MAIN"
  "flights_ithaca_reston:BASE"
  "flights_chi_nyc:BASE"
  "exam:BASE"
  "headphones:BASE"
  "headphones:SOFT"
)

CANONICAL_PAIRS=(
  "flights_ithaca_reston:Complicated"
  "flights_chi_nyc:Complicated_structured"
  "exam:REGISTRAR"
  "headphones:MAIN"
)

SECTION_ORDERS=(
  "persona,attributes,priorities"
  "persona,priorities,attributes"
  "attributes,persona,priorities"
  "attributes,priorities,persona"
  "priorities,persona,attributes"
  "priorities,attributes,persona"
)

DEFAULT_ORDER="attributes,priorities,persona"

GROUP_STAMP="$(date -u +"%Y%m%d_%H%M%S")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/order_sensitivity__REPS${TARGET_REPS}__iters${ITERS}__seed${BASE_SEED}__${GROUP_STAMP}}"
mkdir -p "${OUTPUT_ROOT}"
echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"

run_one() {
  local algo="$1"
  local scenario="$2"
  local mode="$3"
  local api_model="$4"
  local seed_offset="$5"
  local batch_size="${6:-}"
  local prompt_variant="${7:-}"
  local section_order="${8:-}"
  local seed=$((BASE_SEED + seed_offset))

  local cmd_args=(
    --algo "${algo}"
    --scenario "${scenario}"
    --mode "${mode}"
    --api-model "${api_model}"
    --iterations "${ITERS}"
    --seed "${seed}"
    --output-root "${OUTPUT_ROOT}/${scenario}"
    --run-stamp "${GROUP_STAMP}"
  )

  if [[ -n "${batch_size}" ]]; then
    cmd_args+=(--batch-size "${batch_size}")
  fi

  if [[ -n "${prompt_variant}" ]]; then
    case "${algo}" in
      utility)
        cmd_args+=(--utility-prompt-strategy fixed --utility-prompt-variant "${prompt_variant}")
        ;;
      tournament|full_batch)
        cmd_args+=(--comparison-prompt-strategy fixed --comparison-prompt-variant "${prompt_variant}")
        ;;
    esac
  fi

  if [[ -n "${section_order}" ]]; then
    cmd_args+=(--section-order "${section_order}")
  fi

  echo "[RUN] algo=${algo} scen=${scenario} mode=${mode} api=${api_model} seed=${seed}${batch_size:+ B=${batch_size}}${prompt_variant:+ fmt=${prompt_variant}}${section_order:+ order=${section_order}}" >&2
  "${PYTHON_BIN}" "${RUN_ALGO}" "${cmd_args[@]}"
}

missing_seed_offsets() {
  local algo="$1" scenario="$2" mode="$3" api_model="$4"
  local batch_size="${5:-}" prompt_variant="${6:-}" section_order="${7:-}" target_reps="$8"
  ALGO_F="${algo}" SCEN_F="${scenario}" MODE_F="${mode}" API_F="${api_model}" \
  BS_F="${batch_size}" PV_F="${prompt_variant}" SO_F="${section_order}" \
  OUTPUT_ROOT_F="${OUTPUT_ROOT}" BASE_SEED_F="${BASE_SEED}" TARGET_REPS_F="${target_reps}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT_F"]) / os.environ["SCEN_F"]
algo = os.environ["ALGO_F"]
scenario = os.environ["SCEN_F"]
mode = os.environ["MODE_F"]
api_model = os.environ["API_F"]
batch_size = os.environ.get("BS_F", "") or ""
prompt_variant = os.environ.get("PV_F", "") or ""
section_order = os.environ.get("SO_F", "") or ""
base_seed = int(os.environ["BASE_SEED_F"])
target_reps = int(os.environ["TARGET_REPS_F"])
section_order_list = (
    [part.strip().lower() for part in section_order.split(",") if part.strip()]
    if section_order
    else None
)

matches = 0
used_offsets: set[int] = set()
if root.is_dir():
    for path in root.glob("*.json"):
        try:
            meta = json.loads(path.read_text()).get("meta", {})
        except Exception:
            continue
        if meta.get("algo") != algo:
            continue
        if meta.get("scenario") != scenario:
            continue
        if meta.get("mode") != mode:
            continue
        if meta.get("api_model") != api_model:
            continue
        if batch_size:
            meta_batch = meta.get("batch_size")
            if meta_batch is None or int(meta_batch) != int(batch_size):
                continue
        cfg = meta.get("config") or {}
        if prompt_variant:
            meta_variant = (
                cfg.get("comparison_prompt_variant_override")
                or cfg.get("utility_prompt_variant_override")
            )
            if meta_variant != prompt_variant:
                continue
        if section_order_list is not None:
            meta_order = cfg.get("section_order")
            if not isinstance(meta_order, list):
                continue
            if [str(part).lower() for part in meta_order] != section_order_list:
                continue

        matches += 1
        seed = meta.get("seed")
        if isinstance(seed, int):
            used_offsets.add(seed - base_seed)

remaining = max(0, target_reps - matches)
offset = 0
while remaining > 0:
    if offset not in used_offsets:
        print(offset)
        remaining -= 1
    offset += 1
PY
}

export -f run_one
export OUTPUT_ROOT GROUP_STAMP RUN_ALGO PYTHON_BIN BASE_SEED ITERS

submit_jobs() {
  local section_name="$1"
  shift
  local jobs=("$@")
  local round=0

  while true; do
    round=$((round + 1))
    local pending=()
    local serial_pending=()
    local all_done=true

    for spec in "${jobs[@]}"; do
      IFS="|" read -r algo scen mode api bs pv so reps <<<"${spec}"
      seed_offsets="$(missing_seed_offsets "${algo}" "${scen}" "${mode}" "${api}" "${bs}" "${pv}" "${so}" "${reps}")"
      if [[ -z "${seed_offsets}" ]]; then
        continue
      fi

      all_done=false
      remaining="$(printf '%s\n' "${seed_offsets}" | sed '/^$/d' | wc -l | tr -d ' ')"
      current=$((reps - remaining))
      echo "  [${section_name} R${round}] ${algo}/${scen}/${mode}/${api}${bs:+/B${bs}}${pv:+/${pv}}${so:+/[${so}]}: ${current}/${reps} queued ${remaining}"

      while IFS= read -r seed_offset; do
        [[ -z "${seed_offset}" ]] && continue
        job="${algo}|${scen}|${mode}|${api}|${bs}|${pv}|${so}|${seed_offset}"
        if [[ "${algo}" == "full_batch" ]]; then
          serial_pending+=("${job}")
        else
          pending+=("${job}")
        fi
      done <<<"${seed_offsets}"
    done

    ${all_done} && break
    if [[ ${round} -gt ${MAX_ROUNDS} ]]; then
      echo "[WARN] ${section_name}: hit MAX_ROUNDS=${MAX_ROUNDS}; some jobs may not have reached target"
      break
    fi

    if [[ ${#pending[@]} -gt 0 ]]; then
      printf '%s\n' "${pending[@]}" | xargs -n1 -P"${JOBS}" bash -c '
        IFS="|" read -r algo scen mode api bs pv so seed_off <<<"$0"
        run_one "${algo}" "${scen}" "${mode}" "${api}" "${seed_off}" "${bs}" "${pv}" "${so}" \
          || echo "[WARN] job failed: ${algo}/${scen}/${mode}/${api}/B${bs}/${pv}/[${so}]/seed${seed_off}"
      '
    fi

    if [[ ${#serial_pending[@]} -gt 0 ]]; then
      for job in "${serial_pending[@]}"; do
        IFS="|" read -r algo scen mode api bs pv so seed_off <<<"${job}"
        run_one "${algo}" "${scen}" "${mode}" "${api}" "${seed_off}" "${bs}" "${pv}" "${so}" \
          || echo "[WARN] job failed: ${algo}/${scen}/${mode}/${api}/B${bs}/${pv}/[${so}]/seed${seed_off}"
      done
    fi
  done
}

echo "=== [SECTION 0] Utility order sweep, baselines, and full-batch ==="
S0_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${ALL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    for so in "${SECTION_ORDERS[@]}"; do
      S0_JOBS+=("utility|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${so}|${TARGET_REPS}")
    done
    S0_JOBS+=("baseline|${scen}|${mode}|${api}||||${BASELINE_REPS}")
    S0_JOBS+=("full_batch|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${DEFAULT_ORDER}|${TARGET_REPS}")
  done
done
submit_jobs "S0" "${S0_JOBS[@]}"

echo "[dedupe] Keeping one z-score baseline per scenario/mode/api group"
"${PYTHON_BIN}" "${SCRIPT_DIR}/dedupe_zscore.py" "${OUTPUT_ROOT}" || \
  echo "[WARN] dedupe_zscore.py failed; continuing"

echo "=== [SECTION 1] Tournament section-order sweep @ B=${MAIN_BATCH_SIZE} ==="
S1_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${ALL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    for so in "${SECTION_ORDERS[@]}"; do
      S1_JOBS+=("tournament|${scen}|${mode}|${api}|${MAIN_BATCH_SIZE}|${DEFAULT_PROMPT}|${so}|${TARGET_REPS}")
    done
  done
done
submit_jobs "S1" "${S1_JOBS[@]}"

echo "=== [SECTION 2] Tournament batch-size sweep, default section order ==="
S2_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${CANONICAL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    for bs in "${SWEEP_BATCH_SIZES[@]}"; do
      S2_JOBS+=("tournament|${scen}|${mode}|${api}|${bs}|${DEFAULT_PROMPT}|${DEFAULT_ORDER}|${TARGET_REPS}")
    done
  done
done
submit_jobs "S2" "${S2_JOBS[@]}"

echo "[DONE] Runs complete in ${OUTPUT_ROOT}"

if [[ "${RUN_PLOTS}" == "1" ]]; then
  echo "=== [PLOTS] Generating order-sensitivity plots ==="
  PLOT_API_MODEL="${API_MODEL:-${API_MODEL_LIST// /,}}"
  OUTPUT_ROOT="${OUTPUT_ROOT}" PLOT_DIR="${REPO_ROOT}/outputs/plots" API_MODEL="${PLOT_API_MODEL}" \
    bash "${SCRIPT_DIR}/make_plots.sh"
fi
