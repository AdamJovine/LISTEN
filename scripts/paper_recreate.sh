#!/usr/bin/env bash
# Section-order sweep + batch-size sweep.
#
# Layout: every run is written to <OUTPUT_ROOT>/<scenario>/<file>.json so each
# scenario folder contains every batch-size / section-order combination for
# that scenario. Sections distinguish runs by metadata, not folder.
#
# Sections (run in order; resume-by-meta dedups completed work):
#   0. Non-tournament: utility + baseline + full_batch, 9 pairs * 40 reps per api.
#      Groq + gemini lanes execute in parallel.
#   1. Tournament section-order sweep @ B=32, 6 orders * 9 pairs * 40 reps per api.
#   2. Tournament batch-size sweep, 4 canonical pairs * {2,4,8,16,32} * 40 reps per api.
#
# Section 0 runs before tournament so non-tournament series populate fast.
# JOBS controls concurrency *within* each API lane (default 1); lanes for
# distinct api_models always run concurrently regardless of JOBS.
#
# Usage:
#   bash scripts/paper_recreate.sh
#
# Optional env overrides:
#   TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 PYTHON_BIN=python
#   OUTPUT_ROOT=<path>     # resume into an existing run dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ALGO="${REPO_ROOT}/run_algorithm.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

# ── Knobs ───────────────────────────────────────────────────────────────────
TARGET_REPS="${TARGET_REPS:-40}"
BASELINE_REPS="${BASELINE_REPS:-${TARGET_REPS}}"
ITERS="${ITERS:-25}"
BASE_SEED="${BASE_SEED:-1234}"
JOBS="${JOBS:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"

DEFAULT_PROMPT="header_then_task_v1"
MAIN_BATCH_SIZE=32
SWEEP_BATCH_SIZES=(2 4 8 16 32)

API_MODELS=("groq" "gemini")

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
DEFAULT_ORDER="persona,attributes,priorities"

GROUP_STAMP="$(date -u +"%Y%m%d_%H%M%S")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper__REPS${TARGET_REPS}__iters${ITERS}__seed${BASE_SEED}__${GROUP_STAMP}}"
mkdir -p "${OUTPUT_ROOT}"
echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"

# ── Run helper ──────────────────────────────────────────────────────────────
# Args: algo scenario mode api_model seed_offset batch_size prompt_variant section_order
# (batch_size, prompt_variant, and section_order may be empty strings.)
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

# Count completed runs matching all criteria. Reads <OUTPUT_ROOT>/<scenario>/*.json
# and filters by metadata. batch_size, prompt_variant, and section_order args may be empty (no filter).
count_runs() {
  local algo="$1" scenario="$2" mode="$3" api_model="$4"
  local batch_size="${5:-}" prompt_variant="${6:-}" section_order="${7:-}"
  ALGO_F="${algo}" SCEN_F="${scenario}" MODE_F="${mode}" API_F="${api_model}" \
  BS_F="${batch_size}" PV_F="${prompt_variant}" SO_F="${section_order}" OUTPUT_ROOT_F="${OUTPUT_ROOT}" \
  "${PYTHON_BIN}" - <<'EOF'
import json, os, sys
from pathlib import Path
root = Path(os.environ["OUTPUT_ROOT_F"]) / os.environ["SCEN_F"]
algo = os.environ["ALGO_F"]
scen = os.environ["SCEN_F"]
mode = os.environ["MODE_F"]
api  = os.environ["API_F"]
bs   = os.environ.get("BS_F", "") or ""
pv   = os.environ.get("PV_F", "") or ""
so   = os.environ.get("SO_F", "") or ""
so_list = [s.strip().lower() for s in so.split(",") if s.strip()] if so else None
if not root.is_dir():
    print(0); sys.exit()
n = 0
for p in root.glob("*.json"):
    try:
        meta = json.loads(p.read_text()).get("meta", {})
    except Exception:
        continue
    if meta.get("algo") != algo: continue
    if meta.get("scenario") != scen: continue
    if meta.get("mode") != mode: continue
    if meta.get("api_model") != api: continue
    if bs:
        bsm = meta.get("batch_size")
        if bsm is None or int(bsm) != int(bs): continue
    cfg = meta.get("config") or {}
    if pv:
        meta_pv = cfg.get("comparison_prompt_variant_override") or cfg.get("utility_prompt_variant_override")
        if meta_pv != pv: continue
    if so_list is not None:
        meta_so = cfg.get("section_order")
        if not isinstance(meta_so, list): continue
        if [str(x).lower() for x in meta_so] != so_list: continue
    n += 1
print(n)
EOF
}

export -f run_one
export OUTPUT_ROOT GROUP_STAMP RUN_ALGO PYTHON_BIN BASE_SEED ITERS

# Submit a list of jobs and retry until each reaches its target rep count.
# Each spec is "algo|scenario|mode|api|batch_size|prompt_variant|section_order|target_reps".
#
# Jobs are partitioned by api_model into per-API lanes. Each lane runs at
# concurrency JOBS (default 1) and the lanes run *in parallel*, so groq and
# gemini progress simultaneously while each respects its own rate limit.
BASE_SEED_OFFSET=0
submit_jobs() {
  local section_name="$1"; shift
  local jobs=("$@")
  local round=0
  while true; do
    round=$((round + 1))
    declare -A pending_by_api=()
    local serial_pending=()
    local all_done=true

    for spec in "${jobs[@]}"; do
      IFS="|" read -r algo scen mode api bs pv so reps <<<"${spec}"
      local current
      current=$(count_runs "${algo}" "${scen}" "${mode}" "${api}" "${bs}" "${pv}" "${so}")
      local remaining=$((reps - current))
      [[ ${remaining} -le 0 ]] && continue
      all_done=false
      echo "  [${section_name} R${round}] ${algo}/${scen}/${mode}/${api}${bs:+/B${bs}}${pv:+/${pv}}${so:+/[${so}]}: ${current}/${reps} — queuing ${remaining}"
      for ((i=0; i<remaining; i++)); do
        local seed_offset=${BASE_SEED_OFFSET}
        BASE_SEED_OFFSET=$((BASE_SEED_OFFSET + 1))
        local job="${algo}|${scen}|${mode}|${api}|${bs}|${pv}|${so}|${seed_offset}"
        if [[ "${algo}" == "full_batch" ]]; then
          serial_pending+=("${job}")
        else
          pending_by_api[$api]+="${job}"$'\n'
        fi
      done
    done

    ${all_done} && break
    if [[ ${round} -gt ${MAX_ROUNDS} ]]; then
      echo "[WARN] ${section_name}: hit MAX_ROUNDS=${MAX_ROUNDS}; some jobs may not have reached target"
      break
    fi

    local pids=()
    for api in "${!pending_by_api[@]}"; do
      local payload="${pending_by_api[$api]}"
      [[ -z "${payload}" ]] && continue
      (
        printf '%s' "${payload}" | xargs -n1 -P"${JOBS}" bash -c '
          IFS="|" read -r algo scen mode api bs pv so seed_off <<<"$0"
          run_one "${algo}" "${scen}" "${mode}" "${api}" "${seed_off}" "${bs}" "${pv}" "${so}" \
            || echo "[WARN] job failed: ${algo}/${scen}/${mode}/${api}/B${bs}/${pv}/[${so}]/seed${seed_off}"
        '
      ) &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done

    if [[ ${#serial_pending[@]} -gt 0 ]]; then
      for job in "${serial_pending[@]}"; do
        IFS="|" read -r algo scen mode api bs pv so seed_off <<<"${job}"
        run_one "${algo}" "${scen}" "${mode}" "${api}" "${seed_off}" "${bs}" "${pv}" "${so}" \
          || echo "[WARN] job failed: ${algo}/${scen}/${mode}/${api}/B${bs}/${pv}/[${so}]/seed${seed_off}"
      done
    fi
  done
}

# ─── Section 0: Non-tournament (utility / baseline / full_batch) ────────────
# Runs first; groq and gemini lanes execute in parallel (per-API rate limits
# are independent) so LISTEN-U / baseline / full_batch series populate quickly.
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 0] Non-tournament: utility + baseline + full_batch, 9 pairs × ${TARGET_REPS} reps × 2 apis"
echo "═══════════════════════════════════════════════════════════════════"
S0_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${ALL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    S0_JOBS+=("utility|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${DEFAULT_ORDER}|${TARGET_REPS}")
    S0_JOBS+=("baseline|${scen}|${mode}|${api}||||${BASELINE_REPS}")
    S0_JOBS+=("full_batch|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${DEFAULT_ORDER}|${TARGET_REPS}")
  done
done
submit_jobs "S0" "${S0_JOBS[@]}"

# ─── Section 1: Section-order sweep at B=32 ─────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 1] Section-order sweep: tournament @ B=${MAIN_BATCH_SIZE}, 6 orders × 9 pairs × ${TARGET_REPS} reps"
echo "═══════════════════════════════════════════════════════════════════"
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

# ─── Section 2: Batch-size sweep with default ordering ──────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 2] Batch-size sweep: tournament, order=[${DEFAULT_ORDER}], 4 canonical pairs × 5 sizes × ${TARGET_REPS} reps"
echo "═══════════════════════════════════════════════════════════════════"
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

echo "═══════════════════════════════════════════════════════════════════"
echo "[DONE] All runs complete in ${OUTPUT_ROOT}"
echo "═══════════════════════════════════════════════════════════════════"
