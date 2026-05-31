---
license: mit
language:
  - en
pretty_name: LISTEN Multi-Objective Selection Benchmark
size_categories:
  - 1K<n<10K
task_categories:
  - other
tags:
  - llm
  - preference-elicitation
  - multi-objective
  - benchmark
  - decision-making
configs:
  - config_name: exam
    data_files: exam_data.csv
  - config_name: flights_chi_nyc
    data_files: "Chicago_New York City_combined_numeric_filtered.csv"
  - config_name: flights_ithaca_reston
    data_files: "Leg 1 Ithaca to Reston VA_numeric.csv"
  - config_name: headphones
    data_files: headphones_data.csv
  - config_name: rerank_h1
    data_files: "rerank_h1/*.csv"
  - config_name: rerank_h2
    data_files: "rerank_h2/*.csv"
  - config_name: rerank_h3
    data_files: "rerank_h3/*.csv"
  - config_name: rerank_h4
    data_files: "rerank_h4/*.csv"
  - config_name: rerank_h5
    data_files: "rerank_h5/*.csv"
---

# LISTEN Multi-Objective Selection Benchmark

Datasets accompanying the paper **"LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection"** (IJCAI-ECAI 2026).

- 📄 Paper: https://huggingface.co/papers/2510.25799
- 💻 Code: https://github.com/AdamJovine/LISTEN

## Overview

LISTEN is a benchmark for evaluating how well an LLM can elicit a user's preferences and select a top option from a large candidate set with many numeric and categorical attributes. The benchmark contains **four scenarios**, each with a candidate set, a documented preference statement, and a small human-curated set of acceptable "winning" rows (`human_sol`). It also contains **five independent human re-rankings** of LLM-produced top-20 shortlists, used in the paper to measure agreement between LISTEN and human raters.

## Scenarios

| Config name              | Scenario                | n candidates | n ranked (top-K) | Description |
|--------------------------|-------------------------|-------------:|------------------:|-------------|
| `exam`                   | Exam Scheduling         | 4,938        | 20                | Cornell final-exam schedules scored on conflict, clustering, and lateness penalties. |
| `flights_chi_nyc`        | Flights CHI→NYC         | 903          | 20                | Filtered Google Flights itineraries with structured layover / duration / price columns. |
| `flights_ithaca_reston`  | Flights Ithaca→Reston   | 216          | 20                | Smaller multi-leg search; more varied carrier / stop combinations. |
| `headphones`             | Headphones              | 77           | 20                | Amazon product listings with technical specs (driver size, battery, ANC, reviews). |

Loading a scenario:

```python
from datasets import load_dataset

exam = load_dataset("AdamJovine/LISTEN-benchmark", "exam", split="train")
flights = load_dataset("AdamJovine/LISTEN-benchmark", "flights_chi_nyc", split="train")
headphones = load_dataset("AdamJovine/LISTEN-benchmark", "headphones", split="train")
```

## Human re-rankings (`rerank_h1` … `rerank_h5`)

Five different humans (h1–h5) were each asked to re-rank the top-20 shortlist produced by LISTEN on a subset of scenarios. Each rerank CSV adds two leading columns to the scenario schema:

- `rank` — the human's 1-best ordering (1 = preferred).
- `_row_id` — the original row index in the full candidate CSV.

Coverage per annotator (not every annotator did every scenario):

|        | exam | flights_chi_nyc | flights_ithaca_reston | headphones |
|--------|:----:|:---------------:|:---------------------:|:----------:|
| h1     |      | ✓               | ✓                     | ✓          |
| h2     | ✓    |                 | ✓                     | ✓          |
| h3     | ✓    | ✓               | ✓                     |            |
| h4     | ✓    | ✓               | ✓                     | ✓          |
| h5     | ✓    | ✓               |                       | ✓          |

Loading a rerank:

```python
rerank = load_dataset("AdamJovine/LISTEN-benchmark", "rerank_h4", split="train")
```

## Schema notes

- **`exam_data.csv`** — numeric penalty/weight columns (`conflicts`, `quints`, `quads`, `triple in 24h (no gaps)`, `lateness`, `avg_max`, …) plus the schedule's hyperparameters (`alpha`, `gamma`, `delta`, `vega`, `theta`, …).
- **Flights CSVs** — flight itineraries with `origin`, `destination`, `departure time`, `arrival time`, `duration`, `stops`, `price`, distance-from-airport columns, and (for `flights_chi_nyc`) a derived `meets_requirements` flag.
- **`headphones_data.csv`** — Amazon product attributes: `asin`, `product_name`, `brand` (encoded), `price`, `type`, `connectivity`, `noise_cancellation`, `battery_life`, `bluetooth_version`, `driver_size`, `weight`, `water_resistance`, `microphone`, `review_rating`, `review_count`, `description`.

The complete preference statements used in the paper (and the `human_sol` IDs for each scenario) live alongside the scenario configs in the GitHub repo under [`configs/`](https://github.com/AdamJovine/LISTEN/tree/main/configs).

## How the benchmark is used

In the paper, an LLM is given (a) a preference statement and (b) the candidate set, and is asked to return a top-K shortlist. The primary evaluation metric is:

- **NAR (Normalized Average Rank)** — the average rank of the LLM-selected items under each human re-ranking, normalized to `[0, 1]` so that lower is better (rank 1 = the human's most preferred item).

A complementary dataset diagnostic, **concordance** (fraction of random linear utility functions whose argmax lands in `human_sol`, paper §4.1), is implemented in [`post_analysis/concordance_analysis.py`](https://github.com/AdamJovine/LISTEN/blob/main/post_analysis/concordance_analysis.py) in the GitHub repository.

## License

[MIT](https://github.com/AdamJovine/LISTEN/blob/main/LICENSE). The `headphones_data.csv` `description` column contains short product blurbs scraped from public Amazon listings; redistributed under fair-use for academic benchmarking.

## Citation

```bibtex
@inproceedings{jovine2026listen,
  title={LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection},
  author={Jovine, Adam S and Ye, Tinghan and Bahk, Francis and Wang, Jingjing and Ford, Matthew and Shmoys, David B and Frazier, Peter I},
  booktitle={Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year={2026}
}
```
