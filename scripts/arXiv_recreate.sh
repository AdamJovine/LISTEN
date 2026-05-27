#!/usr/bin/env bash
# Full arXiv pipeline (self-contained).
#
# Stage 1 — experiments. Written to <OUTPUT_ROOT>/<scenario>/<file>.json; runs
# distinguish themselves by metadata, not folder. Resume-by-meta dedups
# completed work.
#   Section 0: utility (6 orders) + baseline + full_batch, 9 pairs × 2 apis.
#   Section 1: tournament section-order sweep @ B=32, 6 orders × 9 pairs × 2 apis.
#   Section 2: tournament batch-size sweep, 4 canonical pairs × {2,4,8,16,32} × 2 apis.
# Section 0 runs first so non-tournament series populate fast. JOBS controls
# concurrency *within* each API lane (default 1); groq/gemini lanes always run
# concurrently regardless of JOBS.
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
# Optional env overrides:
#   TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 PYTHON_BIN=python
#   OUTPUT_ROOT=<path>     # resume into an existing run dir
#   PLOT_DIR=<path>        # override plot destination (default outputs/plots)
#   SKIP_RUNS=1            # only do Stage 2 (assumes OUTPUT_ROOT already populated)

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
MAIN_BATCH_SIZE="${MAIN_BATCH_SIZE:-32}"
HEADPHONES_BATCH_SIZE="${HEADPHONES_BATCH_SIZE:-8}"
SWEEP_BATCH_SIZES=(2 4 8 16 32)
DEFAULT_ORDER="${DEFAULT_ORDER:-attributes,priorities,persona}"

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

GROUP_STAMP="${GROUP_STAMP:-$(date -u +"%Y%m%d_%H%M%S")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper__REPS${TARGET_REPS}__iters${ITERS}__seed${BASE_SEED}__${GROUP_STAMP}}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/outputs/plots}"
mkdir -p "${OUTPUT_ROOT}" "${PLOT_DIR}"

echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"
echo "[PLOT_DIR]    ${PLOT_DIR}"
echo

# ── Helpers ─────────────────────────────────────────────────────────────────

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

# Args: algo scenario mode api_model seed_offset batch_size prompt_variant section_order
# (batch_size, prompt_variant, and section_order may be empty strings.)
run_one() {
  local algo="$1" scenario="$2" mode="$3" api_model="$4"
  local seed_offset="$5" batch_size="${6:-}" prompt_variant="${7:-}" section_order="${8:-}"
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
# and filters by metadata. batch_size, prompt_variant, section_order may be empty.
count_runs() {
  local algo="$1" scenario="$2" mode="$3" api_model="$4"
  local batch_size="${5:-}" prompt_variant="${6:-}" section_order="${7:-}"
  ALGO_F="${algo}" SCEN_F="${scenario}" MODE_F="${mode}" API_F="${api_model}" \
  BS_F="${batch_size}" PV_F="${prompt_variant}" SO_F="${section_order}" OUTPUT_ROOT_F="${OUTPUT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
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
PY
}

# Zscore-avg is deterministic — every baseline run for the same
# (scenario, mode, api_model) computes the same winner. Keep zscore_winner
# in the oldest baseline JSON per group and null it out elsewhere so
# expand_baseline_variants emits exactly one BaselineZscore entry per group.
# Idempotent.
dedupe_zscore() {
  OUTPUT_ROOT_F="${OUTPUT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json, os, sys
from collections import defaultdict
from pathlib import Path
root = Path(os.environ["OUTPUT_ROOT_F"])
if not root.is_dir():
    print(f"[ERR] not a directory: {root}", file=sys.stderr)
    sys.exit(1)
groups = defaultdict(list)
for path in sorted(root.glob("**/*.json")):
    if path.name in ("run_info.json", "manifest.json"):
        continue
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    meta = data.get("meta", {})
    if meta.get("algo") != "baseline":
        continue
    key = (meta.get("scenario"), meta.get("mode"), meta.get("api_model"))
    groups[key].append(path)
total_changed = 0
kept_total = 0
for key, paths in sorted(groups.items()):
    paths.sort(key=lambda p: p.stat().st_mtime)
    keeper = paths[0]
    kept_total += 1
    cleared = 0
    for p in paths[1:]:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        history = (payload.get("results") or {}).get("history")
        if not isinstance(history, dict):
            continue
        batch_comps = history.get("batch_comparisons")
        if not isinstance(batch_comps, list):
            continue
        modified = False
        for entry in batch_comps:
            if isinstance(entry, dict) and entry.get("zscore_winner") is not None:
                entry["zscore_winner"] = None
                modified = True
        if modified:
            p.write_text(json.dumps(payload, indent=2, default=str))
            cleared += 1
            total_changed += 1
    print(f"  {key}: kept {keeper.name} | nulled zscore_winner in {cleared} of {len(paths)-1} others")
print(f"[DONE] {kept_total} groups kept, {total_changed} files modified.")
PY
}

export -f run_one
export OUTPUT_ROOT GROUP_STAMP RUN_ALGO PYTHON_BIN BASE_SEED ITERS

# Submit a list of jobs and retry until each reaches its target rep count.
# Each spec is "algo|scenario|mode|api|batch_size|prompt_variant|section_order|target_reps".
# Jobs are partitioned by api_model into per-API lanes; lanes run concurrently
# while each respects its own rate limit. Within a lane, concurrency=JOBS.
#
# Seed assignment is per metadata cell: a cell that already has `current`
# reps on disk gets seed offsets [current, current+1, ...]. This keeps resumes
# from colliding with existing filenames for the same cell.
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
        local seed_offset=$((current + i))
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

# ── Stage 1: experiments ───────────────────────────────────────────────────
if [[ "${SKIP_RUNS:-0}" != "1" ]]; then

  echo "═══════════════════════════════════════════════════════════════════"
  echo "[SECTION 0] Non-tournament: utility (6 orders) + baseline + full_batch, 9 pairs × ${TARGET_REPS} reps × 2 apis"
  echo "═══════════════════════════════════════════════════════════════════"
  S0_JOBS=()
  for api in "${API_MODELS[@]}"; do
    for pair in "${ALL_PAIRS[@]}"; do
      IFS=":" read -r scen mode <<<"${pair}"
      # Utility sweeps all 6 section orders so LISTEN-U has matching
      # order-by-order data for the cross-scenario / order plots.
      for so in "${SECTION_ORDERS[@]}"; do
        S0_JOBS+=("utility|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${so}|${TARGET_REPS}")
      done
      # Baseline samples BaselineRandom; zscore_winner is deterministic and
      # gets deduped after the section completes. Baseline is prompt-agnostic
      # so it does not vary by section_order.
      S0_JOBS+=("baseline|${scen}|${mode}|${api}||||${BASELINE_REPS}")
      S0_JOBS+=("full_batch|${scen}|${mode}|${api}||${DEFAULT_PROMPT}|${DEFAULT_ORDER}|${TARGET_REPS}")
    done
  done
  submit_jobs "S0" "${S0_JOBS[@]}"

  echo "[dedupe] nulling duplicate zscore_winner entries in baseline JSONs"
  dedupe_zscore || echo "[WARN] dedupe_zscore failed (continuing)"

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

else
  echo "[STAGE 1] SKIP_RUNS=1 — skipping experiments"
fi

# ── Stage 2: plots ─────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════"
echo "[STAGE 2] Generating plots into ${PLOT_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

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
