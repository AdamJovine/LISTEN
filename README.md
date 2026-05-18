# LISTEN — Reproducibility code

This repository contains the code and configuration files needed to reproduce
the experiments and figures in the LISTEN paper. Two LLM APIs are supported:
**Groq** (Llama 3.3 70B) and **Google Gemini** (default model:
`gemini-2.5-flash-lite`). Four
algorithms are included: `tournament` (LISTEN-T), `utility` (LISTEN-U),
`baseline` (random + z-score variants), and `full_batch`.

## TL;DR — reproducing the paper

After setting up the environment (§2) and API keys (§3), the **entire** paper
is reproduced by running one of two top-level driver scripts from the
repository root:

```bash
bash scripts/arXiv_recreate.sh    # arXiv version: all runs + all plots
bash scripts/IJCAI_recreate.sh    # IJCAI submission: same runs, IJCAI plot subset
```

Each script runs all experiments and then generates the corresponding paper
plots. No other commands are needed. Both scripts are resume-safe: they read
existing run JSONs in `OUTPUT_ROOT`, count completed reps per cell, and only
launch what's missing. See §5 for knobs (`TARGET_REPS`, `JOBS`,
`OUTPUT_ROOT`, …).

Both scripts are Linux/macOS only (they use bash-only features like
`xargs -P`); on Windows, run them under WSL.

## 1) Repository layout

```
.
├── run_algorithm.py        # CLI: single algorithm run
├── experiment.py           # Base class + iteration loop
├── iteration.py            # Per-iteration result dataclass
├── *_algorithm.py          # tournament, utility, baseline, full_batch
├── base_client.py          # Abstract LLM client
├── groq_client.py          # Groq API client
├── gemini_client.py        # Gemini / Vertex AI client (google.genai SDK)
├── prompt_template.py      # Abstract prompt template
├── prompt_tournament.py    # Comparison prompt (tournament + full_batch)
├── prompt_utility.py       # Weight-elicitation prompt (utility)
├── configs/                # Global + per-scenario YAML configs
├── input/                  # CSV data for each scenario + human-rerank rankings
├── plotting/               # Figure-generation scripts
├── post_analysis/          # Concordance metric (paper §4.1) + sweep helpers
├── scripts/                # arXiv_recreate.sh + IJCAI_recreate.sh
└── tests/                  # pytest suite
```

## 2) Environment

```bash
conda env create -f environment.yml
conda activate listen
```

`environment.yml` is the single source of truth for dependencies. There is
no separate `requirements.txt`; if you need a pip-only install, mirror the
package list from `environment.yml`.

## 3) API keys

Create a `.env` file in the repo root:

```bash
GROQ_API_KEY="<your_groq_key>"
GEMINI_API_KEY="<your_gemini_key>"
GOOGLE_API_KEY="<your_google_key>"   # alias accepted for Gemini
```

Gemini is accessed via the `google-genai` SDK (the legacy
`google-generativeai` package is end-of-life and no longer used). Both
`GEMINI_API_KEY` and `GOOGLE_API_KEY` are honoured. Vertex AI is used only
when logprobs are explicitly requested. The default Gemini model is
`gemini-2.5-flash-lite`.

## 4) Scenarios

Each scenario lives in `configs/<scenario>.yml`. The paper uses these four
scenarios, each with a canonical mode:

| Scenario     | Canonical mode             | Data file                                                |
|--------------|----------------------------|----------------------------------------------------------|
| `exam`       | `REGISTRAR`                | `input/exam_data.csv`                                    |
| `flights_chi_nyc`   | `Complicated_structured`   | `input/Chicago_New York City_combined_numeric_filtered.csv` |
| `flights_ithaca_reston`   | `Complicated`              | `input/Leg 1 Ithaca to Reston VA_numeric.csv`            |
| `headphones` | `MAIN`             | `input/headphones_data.csv`                              |

Each config also defines a `BASE` mode (no preference utterance) used in the
preference-utterance ablation, and `headphones` additionally defines `SOFT`.

Mapping to paper terminology:

| Paper name                  | Code path                                |
|-----------------------------|------------------------------------------|
| Exam Scheduling             | `exam` / `REGISTRAR`                     |
| Flights CHI→NYC             | `flights_chi_nyc` / `Complicated_structured` |
| Flights Ithaca→Reston       | `flights_ithaca_reston` / `Complicated`  |
| Headphones                  | `headphones` / `MAIN`                    |
| Headphones-Soft (§4.5)      | `headphones` / `SOFT`                    |

## 5) Reproducing the paper — script details

`scripts/arXiv_recreate.sh` and `scripts/IJCAI_recreate.sh` share the same
Stage 1 experiment grid and differ only in their Stage 2 plot output.

**Stage 1 — experiments.** Three sections, all resume-by-metadata:

- **Section 0:** `utility` (6 section orders) + `baseline` + `full_batch`, on
  9 (scenario, mode) pairs × 2 APIs.
- **Section 1:** `tournament` section-order sweep at B=32, 6 orders × 9 pairs
  × 2 APIs.
- **Section 2:** `tournament` batch-size sweep B ∈ {2, 4, 8, 16, 32}, 4
  canonical pairs × 2 APIs.

After Section 0 a dedupe pass keeps `zscore_winner` in exactly one baseline
JSON per (scenario, mode, api) group (z-score is deterministic).

**Stage 2 — plots.** Both scripts write into `outputs/plots/`. `arXiv` emits
the full plot set; `IJCAI` emits the subset that appears in the IJCAI
submission.

| Plot                                                                | Script                            | arXiv | IJCAI |
|---------------------------------------------------------------------|-----------------------------------|:-----:|:-----:|
| Per-scenario tournament batch-size sweep                            | `plotting/general_plot.py`        |  ✓    |   ✓   |
| Cross-scenario × algorithm, aggregated over orders (groq + gemini)  | `plotting/plot_orders_by_algo.py` |  ✓    |   ✓   |
| Cross-scenario × algorithm × section order (groq + gemini)          | `plotting/plot_orders_by_algo.py` |  ✓    |       |
| Cross-scenario × algorithm × section order (groq only)              | `plotting/plot_orders_by_algo.py` |       |   ✓   |
| BASE vs canonical preference-utterance ablation                     | `plotting/plot_base_study.py`     |  ✓    |       |
| Headphones LISTEN-T vs LISTEN-U @ B=8 (both APIs)                   | `plotting/headphones_plot.py`     |  ✓    |       |
| Headphones LISTEN-T vs LISTEN-U @ B=8 (groq only)                   | `plotting/headphones_plot.py`     |       |   ✓   |

`plot_orders_by_algo.py` overlays the human-rerank baselines automatically
by reading `input/rerank_*/`.

**Optional env overrides** accepted by both scripts:

```bash
TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 \
OUTPUT_ROOT=outputs/<existing-run>      # set to resume an in-progress run
bash scripts/arXiv_recreate.sh
```

| Variable       | Purpose                                                            |
|----------------|--------------------------------------------------------------------|
| `TARGET_REPS`  | Successful repetitions per cell of the experiment grid             |
| `ITERS`        | LLM calls per run (`--iterations`)                                 |
| `BASE_SEED`    | Per-cell first replicate seed; later reps use `BASE_SEED + i`       |
| `JOBS`         | Per-API-lane concurrency for `xargs -P` (groq and gemini lanes always run in parallel) |
| `OUTPUT_ROOT`  | Existing run directory to resume into (default: timestamped path)  |
| `PLOT_DIR`     | Plot destination (default `outputs/plots`)                         |
| `SKIP_RUNS=1`  | Skip Stage 1 and only re-generate plots from an existing `OUTPUT_ROOT` |

**Output layout** (timestamp/stamp matches the run):

```
outputs/paper__REPS40__iters25__seed1234__<stamp>/
├── exam/                       all exam runs (every algo / mode / batch / order)
├── flights_chi_nyc/            all flights_chi_nyc runs
├── flights_ithaca_reston/      all flights_ithaca_reston runs
└── headphones/                 all headphones runs
outputs/plots/                  generated plots + CSV tables
```

Each script-run JSON is written to
`OUTPUT_ROOT/<scenario>/<scenario>__<algo>__<mode>__api<...>__<...>.json`
containing the winner, NAR, GTU, full iteration history, and a snapshot of
the resolved config.

**Pre-computed sweep outputs.** The exact JSON outputs for the
`paper__REPS40__iters25__seed1234__20260511_163727` sweep cited in the
paper are published as per-scenario gzipped tarballs on the
[`runs-20260511-163727` GitHub Release](https://github.com/AdamJovine/LISTEN/releases/tag/runs-20260511-163727)
(~650 MB compressed, ~7.3 GB and ~11,800 JSONs uncompressed). The
raw JSONs are intentionally not committed to git; the canonical plots
they produce live under `outputs/plots/`. To skip Stage 1 and
regenerate only the plots from the released outputs:

```bash
RUN=outputs/paper__REPS40__iters25__seed1234__20260511_163727
mkdir -p "$RUN" && (cd "$RUN" && \
  for a in exam.tar.gz flights_chi_nyc.tar.gz \
           flights_ithaca_reston.tar.gz headphones.tar.gz; do
    gh release download runs-20260511-163727 --pattern "$a" --clobber
    tar -xzf "$a" && rm "$a"
  done)
OUTPUT_ROOT="$RUN" SKIP_RUNS=1 bash scripts/arXiv_recreate.sh
```

| Asset                          | Size  | Scenario               |
|--------------------------------|-------|------------------------|
| `exam.tar.gz`                  | 152 M | `exam/`                |
| `flights_chi_nyc.tar.gz`       |  84 M | `flights_chi_nyc/`     |
| `flights_ithaca_reston.tar.gz` |  66 M | `flights_ithaca_reston/` |
| `headphones.tar.gz`            | 347 M | `headphones/`          |

## 6) Single run (for ad-hoc experimentation)

If you want to inspect or modify a single configuration without launching
the full sweep:

```bash
python run_algorithm.py \
  --algo tournament \
  --scenario flights_ithaca_reston \
  --mode Complicated \
  --api-model groq \
  --iterations 25 \
  --batch-size 8 \
  --seed 1234
```

Common flags:

| Flag                | Purpose                                                                |
|---------------------|------------------------------------------------------------------------|
| `--algo`            | `tournament`, `utility`, `baseline`, or `full_batch`                   |
| `--scenario`        | Scenario name (matches a file in `configs/`)                           |
| `--mode`            | Scenario mode; falls back to the config's `default_mode`               |
| `--api-model`       | `groq` or `gemini`                                                     |
| `--model-name`      | Override model name from `model_configs`                               |
| `--iterations`      | Total LLM calls                                                        |
| `--batch-size`      | Batch size B (tournament/full_batch)                                   |
| `--seed`            | Random seed                                                            |
| `--section-order`   | Comma-separated section order (e.g. `persona,attributes,priorities`)   |
| `--reasoning`       | Include explicit reasoning in prompts                                  |
| `--use-history`     | Include prompt history across iterations                               |
| `--temperature`     | Override LLM temperature for the active client                         |
| `--output-root`     | Output directory (default `outputs/<scenario>/`)                       |
| `--unique-rank-batch` | Ensure each tournament batch contains at least one `human_sol` index |
| `--comparison-prompt-{strategy,variant,seed}` | Override comparison prompt selection (tournament/full_batch) |
| `--utility-prompt-{strategy,variant,seed}`    | Override utility prompt selection (utility) |
| `--dry-run`         | Print resolved config + would-be output path, then exit (no LLM calls) |

## 7) Tests

```bash
pytest tests/
```

The suite covers the experiment base class, scoring/NAR helpers, JSON
serialization round-trip, and the tournament output format.

## 8) Concordance analysis

The dataset diagnostic from §4.1 of the paper — concordance, the fraction
of random linear utility functions whose argmax lands in the
human-curated `human_sol` set — is implemented in
[`post_analysis/concordance_analysis.py`](post_analysis/concordance_analysis.py).

Per-pair:

```bash
python post_analysis/concordance_analysis.py \
  --scenario headphones --mode MAIN \
  --n-samples 10000 --random-seed 42
```

Full sweep over every (scenario, mode) pair with a `human_sol`:

```bash
bash post_analysis/run_concordance_analysis.sh
# or, single-process:
python post_analysis/concordance_analysis.py --n-samples 10000 --random-seed 42 \
  --output post_analysis/concordance_results_n10000.csv
```

A reference sweep at n=10,000, seed=42 is checked in at
[`post_analysis/concordance_results_n10000.csv`](post_analysis/concordance_results_n10000.csv).

## 9) Citation

If you use this code, please cite the paper:

```bibtex
@article{jovine2025listen,
  title={LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection},
  author={Jovine, Adam S and Ye, Tinghan and Bahk, Francis and Wang, Jingjing and Ford, Matthew and Shmoys, David B and Frazier, Peter I},
  journal={arXiv preprint arXiv:2510.25799},
  year={2025}
}
```

## 10) License

[MIT](LICENSE).
