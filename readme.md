# Schedule Comparison Experiment

This repository contains code to run **dueling bandit optimization experiments** on exam scheduling data using a variety of preference models, acquisition functions, and clients (LLMs, simulated utilities, etc.).

The workflow is built around the `ScheduleComparisonExperiment` class, which evaluates different schedules based on metrics, simulates or queries preference clients, and logs the optimization history.

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python 3.9+ and install required packages:

```bash
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n bo-opt python=3.10
conda activate bo-opt
pip install -r requirements.txt
```

---

### 2. Configure Your Run

All main settings are in the top `CONFIG` dictionary of the script:

```python
CONFIG = {
    # I/O
    "data_csv": "input/data.csv",    # input schedule data
    "log_dir": "logs",               # where logs/results go
    "tag": "noutil",                 # label to identify runs

    # Client / mode
    "use_local_llm": True,           # True -> use local OSS model
    "use_utility_sim": False,        # True -> simulate utility prefs
    "local_model_id": "deepseek",    # id for local LLM

    # Experiment meta
    "model_types": ["logistic"],     # ["logistic", "gp"]
    "acq_functions": ["thompson"],   # ["eubo", "ucb", "thompson", "info_gain"]
    "use_history_values": [True],    # track reasoning history
    "reasoning": [False],
    "num_runs": 50,                  # number of seeds/runs
    "prompt_indices": [0],           # which prompt variants

    # Dueling bandit / batching
    "batch_size": 5,
    "n_batches": 50,
    "n_champions": 5,
    "save_interval": 5,

    # Optional preference prompt
    "utility_prompt_text": "",
}
```

Edit this block to control the experiment.

---

### 3. Input Data

- Provide a CSV at the path set in `CONFIG["data_csv"]`.
- Must include the metrics listed in `CONFIG["metric_columns"]` (rows with missing values are dropped).
- Example metrics:
  - `conflicts`
  - `quints`
  - `quads`
  - `evening/morning b2b`
  - `other b2b`
  - `two in three slots`
  - etc.

---

### 4. Run an Experiment

Simply execute the script:

```bash
python main.py
```

(or whatever your driver script is named — typically `main.py` or `config.py`).

Logs and outputs are written under `logs/` with a unique timestamp + Slurm job ID (if running under Slurm).

---

### 5. Clients

You can choose between:

- **Utility simulator** (`use_utility_sim=True`)  
  Uses fixed weights for metrics.
- **Local LLM** (`use_local_llm=True`)  
  Connects to a locally hosted OSS model via `get_local_client`.
- **Remote LLM (Groq)** (`use_local_llm=False`)  
  Requires `GROQ_API_KEY` in your environment.

---

### 6. Output

Each run generates:

- **Run manifest**: JSON file with config, Slurm info, environment.
- **History CSV**: comparisons and optimizer state across batches.
- **Console logs**: progress and final ranking.

Example completion message:

```
Experiment completed successfully!
Top schedule: 1234
Total comparisons: 250
```

---

## ⚙️ Advanced Notes

- **Slurm Integration**  
  If run via Slurm, job metadata is automatically included in the manifest.
- **Prompt Templates**  
  Controlled via `PromptTemplate`, with options for reasoning history and custom utility instructions.
- **Acquisition Functions**  
  Supported: `eubo`, `ucb`, `thompson`, `info_gain`.

---

## 📂 Project Structure

```
.
├── input/
│   └── data.csv              # schedule data
├── logs/                     # experiment outputs
├── Experiment.py             # experiment class
├── LISTEN/                   # bandit optimizer implementations
├── LLM.py                    # free LLM client wrapper
├── remoteOss.py              # local OSS client discovery
├── prompt.py                 # prompt templates
└── config.py (this script)   # main entrypoint
```

---

## 🔑 Environment Variables

If using a remote LLM client (Groq):

```bash
export GROQ_API_KEY="your_api_key_here"
```

---

## 🧩 PromptTemplate Usage

The `PromptTemplate` class builds prompts for preference collection. It:
- Adds **history** of past decisions if `reasoning_history=True`
- Appends **utility guidance** (from `utility_prompt_text`)
- Supports **minimal or reasoned modes** (`reasoning=False/True`)
- Prints schedules with only the metrics in `CONFIG["metric_columns"]`

Example:

```python
pt = PromptTemplate(
    reasoning_history=True,
    utility_prompt="Prioritize minimizing evening/morning b2b and other b2b.",
    metric_columns=CONFIG["metric_columns"],
    reasoning=False  # minimal mode
)

schedule_a = {"conflicts": 5, "evening/morning b2b": 12, "other b2b": 8}
schedule_b = {"conflicts": 3, "evening/morning b2b": 15, "other b2b": 7}

pt.add_to_history(schedule_a, schedule_b, choice="B", reasoning="Lower conflicts dominate.")

prompt = pt.format(schedule_a, schedule_b)
print(prompt)
```

Downstream, always parse the last line for a final tag:

```
FINAL: A
or
FINAL: B
```

---

## 📝 License

Add your license information here.

---

## 🙌 Acknowledgements

This project builds on:
- Dueling bandit optimization algorithms
- Preference-based Bayesian optimization
- OSS large language models

