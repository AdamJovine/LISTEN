# LISTEN — Reproducibility code

This repository contains the code and configuration files needed to reproduce
the experiments and figures in the LISTEN paper. Two LLM APIs are supported:
**Groq** (Llama 3.3 70B) and **Google Gemini** (via Vertex AI). Four
algorithms are included: `tournament` (LISTEN-T), `utility` (LISTEN-U),
`baseline` (random + z-score variants), and `full_batch`.

## Repository layout

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
├── scripts/                # Bash drivers for paper experiments
├── tools/                  # One-shot helpers (e.g. legacy-output rename)
└── tests/                  # pytest suite
```

## 1) Environment

```bash
conda env create -f environment.yml
conda activate listen
```

`environment.yml` is the single source of truth for dependencies. There is
no separate `requirements.txt`; if you need a pip-only install, mirror the
package list from `environment.yml`.

## 2) API keys

Create a `.env` file in the repo root:

```bash
GROQ_API_KEY="<your_groq_key>"
GEMINI_API_KEY="<your_gemini_key>"
GOOGLE_API_KEY="<your_google_key>"   # alias accepted for Gemini
```

Gemini is accessed via the `google-genai` SDK (the legacy
`google-generativeai` package is end-of-life and no longer used). Both
`GEMINI_API_KEY` and `GOOGLE_API_KEY` are honoured. Vertex AI is used only
when logprobs are explicitly requested.

## 3) Scenarios

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

## 4) Single run

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
| `--reasoning`       | Include explicit reasoning in prompts                                  |
| `--use-history`     | Include prompt history across iterations                               |
| `--temperature`     | Override LLM temperature for the active client                         |
| `--output-root`     | Output directory (default `outputs/<scenario>/`)                       |
| `--unique-rank-batch` | Ensure each tournament batch contains at least one `human_sol` index |
| `--comparison-prompt-{strategy,variant,seed}` | Override comparison prompt selection (tournament/full_batch) |
| `--utility-prompt-{strategy,variant,seed}`    | Override utility prompt selection (utility) |
| `--section-order`   | Comma-separated order for `persona`, `attributes`, `priorities` prompt sections |
| `--dry-run`         | Print resolved config + would-be output path, then exit (no LLM calls) |

Each run writes a single JSON to
`outputs/<scenario>/<scenario>__<algo>__<mode>__api<...>__<...>.json`
containing the winner, NAR, GTU, full iteration history, and a snapshot of
the resolved config.

## 5) Reproducing the paper

Two top-level driver scripts reproduce the published results. Both are
Linux/macOS only (they use bash-only features like `xargs -P` and `mktemp -t`);
on Windows, run them under WSL or invoke `run_algorithm.py` directly per the
§4 example.

| Script                       | Reproduces                                   |
|------------------------------|----------------------------------------------|
| `scripts/IJCAI_recreate.sh`  | The runs and plots for the IJCAI submission. |
| `scripts/arXiv_recreate.sh`  | The runs and plots for the arXiv version.    |

Run either script from the repository root:

```bash
bash scripts/IJCAI_recreate.sh
# or
bash scripts/arXiv_recreate.sh
```

The prompt section-order sensitivity sweep is separate from the paper
reproduction workflow:

```bash
bash scripts/order_sensitivity_recreate.sh
```

That script runs all six permutations of `persona`, `attributes`, and
`priorities`, then writes plots such as
`outputs/plots/groq__nar__scenario__by_algo_orders.png`.
It defaults to Groq-only outputs for the open-source artifact set; set
`API_MODELS=groq,gemini` once the Gemini sweep is complete.

Optional env overrides accepted by both scripts:

```bash
TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 \
OUTPUT_ROOT=outputs/<existing-run>      # set to resume an in-progress run
bash scripts/IJCAI_recreate.sh
```

| Variable       | Purpose                                                            |
|----------------|--------------------------------------------------------------------|
| `TARGET_REPS`  | Successful repetitions per cell of the experiment grid             |
| `ITERS`        | LLM calls per run (`--iterations`)                                 |
| `BASE_SEED`    | Seed for the first replicate; later replicates use `BASE_SEED + i` |
| `JOBS`         | Parallelism for `xargs -P`                                         |
| `OUTPUT_ROOT`  | Existing run directory to resume into (default: timestamped path)  |
| `API_MODELS`   | Section-order sweep models; default `groq`                         |

Output layout (timestamp/stamp matches the run):

```
outputs/paper__REPS40__iters25__seed1234__<stamp>/
├── exam/                       all exam runs (every algo / mode / batch / prompt variant)
├── flights_chi_nyc/            all flights_chi_nyc runs
├── flights_ithaca_reston/      all flights_ithaca_reston runs
├── headphones/                 all headphones runs
└── plots/                      generated plots + CSV tables
```

Each script runs the experiment grid as a sequence of sections, each with a
retry-until-target loop, and then generates the corresponding paper plots
under `outputs/plots/`. Local run directories may still contain scratch
`plots/` folders while experimenting, but the tracked open-source artifact
bundle is the flat `outputs/plots/*.png` and `outputs/plots/*.csv` set.

## 6) Plots

All plot scripts can also be run standalone against any run directory:

| Plot                                         | Script                                       | Reads from              |
|----------------------------------------------|----------------------------------------------|-------------------------|
| Headphones SOFT vs MAIN           | `plotting/headphones_plot.py`                | `--output-dir <RUN>`    |
| Cross-scenario × algorithm (LISTEN-T, LISTEN-U, baseline, full-batch, human rerank) | `plotting/general_plot.py` with `--canonical_mode` | `--path <RUN>` |
| Per-scenario batch-size sweep (B={2,4,8,16,32}) | `plotting/general_plot.py`                | `--path <RUN>`          |
| Section-order sensitivity (persona/attributes/priorities) | `plotting/plot_orders_by_algo.py` (PNG + CSV) | `--data-dir <RUN>` |
| Preference-utterance ablation (canonical vs BASE) | `plotting/plot_base_study.py` (PNG + CSV) | `--data-dir <RUN>`     |

The cross-scenario plot overlays human-rerank baselines automatically by
reading `input/rerank_*/`.

To regenerate only the Groq section-order figure from an existing run:

```bash
python plotting/plot_orders_by_algo.py \
  --data-dir outputs/<run-dir> \
  --output-dir outputs/plots \
  --api-model groq \
  --batch-size 32
```

If the raw JSON runs are absent but the matching summary CSV already exists in
`outputs/plots/`, the same command regenerates the PNG from that CSV.

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
