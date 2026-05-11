#!/usr/bin/env bash
# Reproduce all paper experiments and produce all paper plots.
#
# Layout: every run is written to <OUTPUT_ROOT>/<scenario>/<file>.json so
# every scenario folder contains every algo / mode / batch-size / prompt
# variant for that scenario. Sections distinguish runs by metadata, not
# folder.
#
# Sections:
#   1. Tournament batch sweep (B={2,4,8,16,32}) over the 4 canonical
#      (scenario, mode) pairs in the default prompt format
#      (header_then_task_v1).
#   2. Headphones SOFT mode: tournament @ B=8 + utility, default format.
#   3. utility / baseline / full_batch in canonical modes (default format).
#   4. Reverse prompt format (task_then_header_v1) at B=32 over the 4
#      canonical pairs (tournament + utility) — for the order study.
#   5. BASE preference-utterance ablation at B=32 over the 4 BASE
#      scenario/mode pairs (tournament + utility).
#   6. All plots.
#
# Both LLM APIs (groq + gemini) are run for every section.
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
BASELINE_REPS="${BASELINE_REPS:-40}"  # baseline mixes deterministic z-score with random-pick; match TARGET_REPS so the random arm is properly sampled (per paper §4.2: 40 reps)
ITERS="${ITERS:-25}"
BASE_SEED="${BASE_SEED:-1234}"
JOBS="${JOBS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"

DEFAULT_PROMPT="header_then_task_v1"
REVERSE_PROMPT="task_then_header_v1"
MAIN_BATCH_SIZE=32
SWEEP_BATCH_SIZES=(2 4 8 16 32)
ORDER_BATCH_SIZE=32
BASE_BATCH_SIZE=32

API_MODELS=("groq" "gemini")

CANONICAL_PAIRS=(
  "flights_ithaca_reston:Complicated"
  "flights_chi_nyc:Complicated_structured"
  "exam:REGISTRAR"
  "headphones:MAIN"
)
BASE_PAIRS=(
  "flights_ithaca_reston:BASE"
  "flights_chi_nyc:BASE"
  "exam:BASE"
  "headphones:BASE"
)
HEADPHONES_STUDENT="headphones:SOFT"

GROUP_STAMP="$(date -u +"%Y%m%d_%H%M%S")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper__REPS${TARGET_REPS}__iters${ITERS}__seed${BASE_SEED}__${GROUP_STAMP}}"
mkdir -p "${OUTPUT_ROOT}"
echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"

# ── Run helper ──────────────────────────────────────────────────────────────
# Args: algo scenario mode api_model seed_offset batch_size prompt_variant
# (batch_size and prompt_variant may be empty strings.)
run_one() {
  local algo="$1"
  local scenario="$2"
  local mode="$3"
  local api_model="$4"
  local seed_offset="$5"
  local batch_size="${6:-}"
  local prompt_variant="${7:-}"
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

  echo "[RUN] algo=${algo} scen=${scenario} mode=${mode} api=${api_model} seed=${seed}${batch_size:+ B=${batch_size}}${prompt_variant:+ fmt=${prompt_variant}}" >&2
  "${PYTHON_BIN}" "${RUN_ALGO}" "${cmd_args[@]}"
}

# Count completed runs matching all criteria. Reads <OUTPUT_ROOT>/<scenario>/*.json
# and filters by metadata. batch_size and prompt_variant args may be empty (no filter).
count_runs() {
  local algo="$1" scenario="$2" mode="$3" api_model="$4"
  local batch_size="${5:-}" prompt_variant="${6:-}"
  ALGO_F="${algo}" SCEN_F="${scenario}" MODE_F="${mode}" API_F="${api_model}" \
  BS_F="${batch_size}" PV_F="${prompt_variant}" OUTPUT_ROOT_F="${OUTPUT_ROOT}" \
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
    if pv:
        cfg = meta.get("config") or {}
        meta_pv = cfg.get("comparison_prompt_variant_override") or cfg.get("utility_prompt_variant_override")
        if meta_pv != pv: continue
    n += 1
print(n)
EOF
}

export -f run_one
export OUTPUT_ROOT GROUP_STAMP RUN_ALGO PYTHON_BIN BASE_SEED ITERS

# Submit a list of jobs and retry until each reaches its target rep count.
# Each spec is "algo|scenario|mode|api|batch_size|prompt_variant|target_reps".
#
# Seed assignment is per-job: a job that already has `current` reps on disk
# gets seed offsets [current, current+1, ..., current+remaining-1]. This
# makes resume safe (new reps always pick seeds that don't collide with
# what's already on disk for that specific job). Different jobs may share
# the same seed value, which is fine — they are independent experiments
# with different (algo, scenario, mode, api, batch_size, prompt_variant).
submit_jobs() {
  local section_name="$1"; shift
  local jobs=("$@")
  local round=0
  while true; do
    round=$((round + 1))
    local pending=()
    local serial_pending=()
    local all_done=true

    for spec in "${jobs[@]}"; do
      IFS="|" read -r algo scen mode api bs pv reps <<<"${spec}"
      local current
      current=$(count_runs "${algo}" "${scen}" "${mode}" "${api}" "${bs}" "${pv}")
      local remaining=$((reps - current))
      [[ ${remaining} -le 0 ]] && continue
      all_done=false
      echo "  [${section_name} R${round}] ${algo}/${scen}/${mode}/${api}${bs:+/B${bs}}${pv:+/${pv}}: ${current}/${reps} — queuing ${remaining}"
      for ((i=0; i<remaining; i++)); do
        local seed_offset=$((current + i))
        if [[ "${algo}" == "full_batch" ]]; then
          serial_pending+=("${algo}|${scen}|${mode}|${api}|${bs}|${pv}|${seed_offset}")
        else
          pending+=("${algo}|${scen}|${mode}|${api}|${bs}|${pv}|${seed_offset}")
        fi
      done
    done

    ${all_done} && break
    if [[ ${round} -gt ${MAX_ROUNDS} ]]; then
      echo "[WARN] ${section_name}: hit MAX_ROUNDS=${MAX_ROUNDS}; some jobs may not have reached target"
      break
    fi

    if [[ ${#pending[@]} -gt 0 ]]; then
      printf '%s\n' "${pending[@]}" | xargs -n1 -P"${JOBS}" bash -c '
        IFS="|" read -r algo scen mode api bs pv seed_off <<<"$0"
        run_one "${algo}" "${scen}" "${mode}" "${api}" "${seed_off}" "${bs}" "${pv}" \
          || echo "[WARN] job failed: ${algo}/${scen}/${mode}/${api}/B${bs}/${pv}/seed${seed_off}"
      '
    fi
    if [[ ${#serial_pending[@]} -gt 0 ]]; then
      for job in "${serial_pending[@]}"; do
        IFS="|" read -r algo scen mode api bs pv seed_off <<<"${job}"
        run_one "${algo}" "${scen}" "${mode}" "${api}" "${seed_off}" "${bs}" "${pv}" \
          || echo "[WARN] job failed: ${algo}/${scen}/${mode}/${api}/B${bs}/${pv}/seed${seed_off}"
      done
    fi
  done
}

# ─── Section 1: Tournament batch sweep on canonical scenarios ───────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 1] Tournament batch sweep over canonical scenarios"
echo "═══════════════════════════════════════════════════════════════════"
S1_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${CANONICAL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    for bs in "${SWEEP_BATCH_SIZES[@]}"; do
      S1_JOBS+=("tournament|${scen}|${mode}|${api}|${bs}|${DEFAULT_PROMPT}|${TARGET_REPS}")
    done
  done
done
submit_jobs "S1" "${S1_JOBS[@]}"

# ─── Section 2: Headphones SOFT mode ────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 2] Headphones SOFT (tournament B=${MAIN_BATCH_SIZE}, utility)"
echo "═══════════════════════════════════════════════════════════════════"
IFS=":" read -r hp_scen hp_mode <<<"${HEADPHONES_STUDENT}"
S2_JOBS=()
for api in "${API_MODELS[@]}"; do
  S2_JOBS+=("tournament|${hp_scen}|${hp_mode}|${api}|${MAIN_BATCH_SIZE}|${DEFAULT_PROMPT}|${TARGET_REPS}")
  S2_JOBS+=("utility|${hp_scen}|${hp_mode}|${api}||${DEFAULT_PROMPT}|${TARGET_REPS}")
done
submit_jobs "S2" "${S2_JOBS[@]}"

# ─── Section 3: utility / baseline / full_batch in canonical modes ──────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 3] utility + baseline + full_batch in canonical modes"
echo "═══════════════════════════════════════════════════════════════════"
S3_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${CANONICAL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    S3_JOBS+=("utility|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${TARGET_REPS}")
    S3_JOBS+=("baseline|${scen}|${mode}|${api}|||${BASELINE_REPS}")
    S3_JOBS+=("full_batch|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${TARGET_REPS}")
  done
done
submit_jobs "S3" "${S3_JOBS[@]}"

# ─── Section 4: Reverse prompt format at B=32 ───────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 4] Reverse prompt format (${REVERSE_PROMPT}) at B=${ORDER_BATCH_SIZE}"
echo "═══════════════════════════════════════════════════════════════════"
S4_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${CANONICAL_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    S4_JOBS+=("tournament|${scen}|${mode}|${api}|${ORDER_BATCH_SIZE}|${REVERSE_PROMPT}|${TARGET_REPS}")
    S4_JOBS+=("utility|${scen}|${mode}|${api}||${REVERSE_PROMPT}|${TARGET_REPS}")
  done
done
submit_jobs "S4" "${S4_JOBS[@]}"

# ─── Section 5: BASE preference utterance at B=32 ───────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 5] BASE preference-utterance ablation at B=${BASE_BATCH_SIZE}"
echo "═══════════════════════════════════════════════════════════════════"
S5_JOBS=()
for api in "${API_MODELS[@]}"; do
  for pair in "${BASE_PAIRS[@]}"; do
    IFS=":" read -r scen mode <<<"${pair}"
    S5_JOBS+=("tournament|${scen}|${mode}|${api}|${BASE_BATCH_SIZE}|${DEFAULT_PROMPT}|${TARGET_REPS}")
    S5_JOBS+=("utility|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${TARGET_REPS}")
  done
done
submit_jobs "S5" "${S5_JOBS[@]}"

echo "═══════════════════════════════════════════════════════════════════"
echo "[DONE] All runs complete in ${OUTPUT_ROOT}"
echo "═══════════════════════════════════════════════════════════════════"

# ─── Section 6: Plots ───────────────────────────────────────────────────────
PLOT_DIR="${OUTPUT_ROOT}/plots"
mkdir -p "${PLOT_DIR}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "[SECTION 6] Plots -> ${PLOT_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# Plot 1: Headphones SOFT vs MAIN (one figure per api_model)
echo "[PLOT 1] Headphones SOFT vs MAIN"
for api in "${API_MODELS[@]}"; do
  "${PYTHON_BIN}" "${REPO_ROOT}/plotting/headphones_plot.py" \
    --scenario headphones \
    --output-dir "${OUTPUT_ROOT}" \
    --plot-dir "${PLOT_DIR}" \
    --api-model "${api}" \
    --batch-size "${MAIN_BATCH_SIZE}"
done

# Plot 2: Cross-scenario × algorithms (LISTEN-T at B=MAIN_BATCH_SIZE, LISTEN-U, baseline, full_batch, rerank)
echo "[PLOT 2] Cross-scenario × algorithms"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/general_plot.py" \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large api_model all \
  --x_medium scenario all \
  --x_small algo all \
  --y nar \
  --canonical_mode \
  --per_algo_filter tournament batch_size "${MAIN_BATCH_SIZE}"

# Plot 3: Per-scenario × batch sizes (tournament only)
echo "[PLOT 3] Per-scenario × batch sizes (tournament)"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/general_plot.py" \
  --path "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}" \
  --x_large algo tournament scenario all mode all api_model all \
  --x_medium batch_size all \
  --y nar

# Plot 4: Reorder study (header_then_task_v1 vs task_then_header_v1)
echo "[PLOT 4] Reorder study"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_order_study.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}"

# Plot 5: BASE vs canonical mode (preference-utterance ablation)
echo "[PLOT 5] BASE vs canonical mode"
"${PYTHON_BIN}" "${REPO_ROOT}/plotting/plot_base_study.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${PLOT_DIR}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "[ALL DONE]"
echo "[OUTPUT] ${OUTPUT_ROOT}"
echo "[PLOTS]  ${PLOT_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
