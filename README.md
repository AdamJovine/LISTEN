# LISTEN — Reproducibility code (IJCAI submission)

This repository contains the code and configuration files needed to reproduce
the experiments and figures in our IJCAI paper. Two LLM APIs are supported:
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
├── gemini_client.py        # Gemini / Vertex AI client
├── promptTemplate.py       # Abstract prompt template
├── prompt_tournament.py    # Comparison prompt (tournament + full_batch)
├── prompt_utility.py       # Weight-elicitation prompt (utility)
├── configs/                # Global + per-scenario YAML configs
├── input/                  # CSV data for each scenario + human-rerank rankings
├── plotting/               # Figure-generation scripts
├── scripts/                # Bash drivers for paper experiments
└── tests/                  # pytest suite
```

## 1) Environment

```bash
conda env create -f environment.yml
conda activate listen
```

or with pip:

```bash
pip install -r requirements.txt
```

## 2) API keys

Create a `.env` file in the repo root:

```bash
GROQ_API_KEY="<your_groq_key>"
GEMINI_API_KEY="<your_gemini_key>"
GOOGLE_API_KEY="<your_google_key>"   # alias accepted for Gemini
```

Gemini is read via Google's `generativeai` SDK; both `GEMINI_API_KEY` and
`GOOGLE_API_KEY` are honoured.

## 3) Scenarios

Each scenario lives in `configs/<scenario>.yml`. The paper uses these four,
each with a canonical mode:

| Scenario     | Canonical mode             | Data file                                                |
|--------------|----------------------------|----------------------------------------------------------|
| `exam`       | `REGISTRAR`                | `input/exam_data.csv`                                    |
| `flight00`   | `Complicated_structured`   | `input/Chicago_New York City_combined_numeric_filtered.csv` |
| `flight02`   | `Complicated`              | `input/Leg 1 Ithaca to Reston VA_numeric.csv`            |
| `headphones` | `STUDENT_HARD`             | `input/headphones_data.csv`                              |

Each config also defines a `BASE` mode (no preference utterance) used in the
preference-utterance ablation, and `headphones` additionally defines `STUDENT`.

## 4) Single run

```bash
python run_algorithm.py \
  --algo tournament \
  --scenario flight02 \
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

Each run writes a single JSON to
`outputs/<scenario>/<scenario>__<algo>__<mode>__api<...>__<...>.json`
containing the winner, NAR, GTU, full iteration history, and a snapshot of
the resolved config.

## 5) Reproducing the paper

A single script — `scripts/paper_recreate.sh` — does all runs and produces
all plots, for both LLM APIs (groq and gemini).

```bash
bash scripts/paper_recreate.sh
```

Optional env overrides:

```bash
TARGET_REPS=40 ITERS=25 BASE_SEED=1234 JOBS=4 \
OUTPUT_ROOT=outputs/<existing-run>      # set to resume an in-progress run
bash scripts/paper_recreate.sh
```

Output layout:

```
outputs/paper__REPS40__iters25__seed1234__<stamp>/
├── exam/        all exam runs (every algo / mode / batch / prompt variant)
├── flight00/    all flight00 runs
├── flight02/    all flight02 runs
├── headphones/  all headphones runs
└── plots/       the five paper plots + CSV tables
```

The script runs five sections, each with a retry-until-target loop:

| § | What it runs | Reps |
|---|---|---|
| 1 | Tournament × B={2,4,8,16,32} × {flight02:Complicated, flight00:Complicated_structured, exam:REGISTRAR, headphones:STUDENT_HARD} × {groq, gemini}, default prompt `header_then_task_v1` | 40 each |
| 2 | Tournament @ B=8 + utility × headphones:STUDENT × {groq, gemini} | 40 each |
| 3 | Utility / baseline / full_batch × 4 canonical pairs × {groq, gemini} | 40 each |
| 4 | Tournament @ B=32 + utility × 4 canonical pairs × `task_then_header_v1` × {groq, gemini} (reverse-prompt set for the order study) | 40 each |
| 5 | Tournament @ B=32 + utility × {flight02:BASE, flight00:BASE, exam:BASE, headphones:BASE} × {groq, gemini} (preference-utterance ablation) | 40 each |

Section 6 then generates all five paper plots into `<OUTPUT_ROOT>/plots/`.

## 6) Plots

All plot scripts can also be run standalone against any run directory:

| Plot                                         | Script                                       | Reads from              |
|----------------------------------------------|----------------------------------------------|-------------------------|
| Headphones STUDENT vs STUDENT_HARD           | `plotting/headphones_plot.py`                | `--output-dir <RUN>`    |
| Cross-scenario × algorithm (LISTEN-T, LISTEN-U, baseline, full-batch, human rerank) | `plotting/general_plot.py` with `--canonical_mode` | `--path <RUN>` |
| Per-scenario batch-size sweep (B={2,4,8,16,32}) | `plotting/general_plot.py`                | `--path <RUN>`          |
| Reorder (header_then_task vs task_then_header) | `plotting/plot_order_study.py` (PNG + CSV) | `--data-dir <RUN>`      |
| Preference-utterance ablation (canonical vs BASE) | `plotting/plot_base_study.py` (PNG + CSV) | `--data-dir <RUN>`     |

The cross-scenario plot overlays human-rerank baselines automatically by
reading `input/rerank_*/`.

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
  --scenario headphones --mode STUDENT_HARD \
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
  author={Jovine, Adam S and Ye, Tinghan and Bahk, Francis and Wang, Jingjing and Shmoys, David B and Frazier, Peter I},
  journal={arXiv preprint arXiv:2510.25799},
  year={2025}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided; GitHub
renders it as a "Cite this repository" button on the repo page.

## 10) License

[MIT](LICENSE).
