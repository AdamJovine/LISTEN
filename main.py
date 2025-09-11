# === config.py-ish top section you can tweak ===
from __future__ import annotations
import os, json, glob
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from LLM import FreeLLMPreferenceClient
from oss import LocalGroqLikeClient
from Experiment import ScheduleComparisonExperiment , ScheduleBatchExperiment
from LISTEN import CustomDuelingBanditOptimizer, DuelingBanditOptimizer
from batchPref import BatchPrefLearning
from perfect import SimpleUtilityPreferenceClient
from p1 import SimpleB2BPreferenceClient
#from prompt import PromptTemplate, BoundedPromptTemplate 
from prompt import BeautyPromptTemplateAdapter, SchedulePromptTemplateAdapter 
from remoteOss import get_local_client
batch_size = 64
# ------ CENTRAL CONFIG (edit these) ------
CONFIG = {
    # I/O
    "data_csv": "input/data.csv",
    "log_dir": "logs",
    "tag": "noutil",  # free-form label to help identify runs

    # Client / mode
    "use_local_llm": True,      # True -> use local OSS-20B via get_local_client()
    "use_utility_sim": False,    # True -> simulated utility preferences; False -> LLM preferences
    "local_model_id": "deepseek",

    # Experiment meta
    "pair_mode" : False, 
    "model_types": ["logistic"],              # ["logistic", "gp"]
    "acq_functions": ["eubo"],                # ["eubo","ucb","thompson","info_gain", "random"]
    "use_history_values": [False],             # [True, False]
    'reasoning': [False], 
    "num_runs": 50,                            # how many seeds/runs of each config
    "prompt_indices": [0],                    # which prompt variants to use

    # Dueling bandit / batching
    
    "batch_size": batch_size,
    "n_batches":  25 ,
    "n_champions": 5,
    "save_interval":batch_size ,

    # Prompt/template knobs
    "utility_prompt_text":#'',#"Your first priority is to minimize conflicts, this is the most important metic. Your next priority is to minimize triple exams (triple in 24h (no gaps) and triple in same day (no gaps)), finally minimize back to back exams. ",  
       'Make sure to prioritize minimising back to back exams ( evening/morning b2b / other b2b)" this is the most important metric', #"prioritize minimizing back to back exams (evening/morning b2b) and (other b2b)",
    #

    # CSV metrics to keep (drop rows with NaNs in these)
    "metric_columns": [
        "conflicts",
        "quints",
        "quads",
        "four in five slots",
        "triple in 24h (no gaps)",
        "triple in same day (no gaps)",
        "three in four slots",
        "evening/morning b2b",
        "other b2b",
        "two in three slots",
        "avg_max"
    ],
}
# -----------------------------------------

# ==== Run stamp & paths ====
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", None)
RUN_ID = f"b2bmin_trubatch{RUN_STAMP}batch_size{batch_size}" + (f"_job{SLURM_JOB_ID}" if SLURM_JOB_ID else "")

current_dir = Path(__file__).resolve().parent
log_dir = current_dir / CONFIG["log_dir"]
log_dir.mkdir(parents=True, exist_ok=True)

# Save a manifest ASAP so even early failures leave breadcrumbs
manifest = {
    "run_id": RUN_ID,
    "stamp_utc": RUN_STAMP,
    "tag": CONFIG["tag"],
    "config": CONFIG,
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node_list": os.environ.get("SLURM_NODELIST"),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
    },
    "env": {
        "python": os.environ.get("PYTHON"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    },
}
(log_dir / f"{RUN_ID}_run_manifest.json").write_text(json.dumps(manifest, indent=2))

# ==== Data load & filtering ====
load_dotenv()
data_path = current_dir / CONFIG["data_csv"]
df = pd.read_csv(data_path)
df = df.dropna(subset=CONFIG["metric_columns"])

# ==== Preference client selection ====
if CONFIG["use_utility_sim"]:
    # Example weights; adjust freely or expose to CONFIG
    WEIGHTS = {
        "conflicts": 000,
        "quints": 0,
        "quads": 0,
        "four in five slots": 0,
        "triple in 24h (no gaps)": 0,
        "triple in same day (no gaps)": 0,
        "three in four slots": 0,
        "evening/morning b2b": -1,
        "other b2b": -1,
        "two in three slots": 0,
    }
    client = SimpleUtilityPreferenceClient(weights=WEIGHTS)
else:
    if CONFIG["use_local_llm"]:
        client = get_local_client(CONFIG["local_model_id"])
    else:
        api_key = os.getenv("GROQ_API_KEY")
        client = FreeLLMPreferenceClient(
            provider="groq",
            api_key=api_key,
            model_name="openai/gpt-oss-20b",
            simple=False,
            rate_limit_delay=0.1,
            max_tokens=1024,
        )

# ==== Main loops ====
for model_type in CONFIG["model_types"]:
    for run_idx in range(CONFIG["num_runs"]):
        for prompt_idx in CONFIG["prompt_indices"]:
            for use_history in CONFIG["use_history_values"]:
                # Base filename (no dirs); every artifact gets the unique run prefix
                base = (
                    f"zb2bmax{CONFIG['use_utility_sim']}_"
                    f"loc{CONFIG['use_local_llm']}_"
                    f"{model_type}_"
                    f"acq{{acq}}_"
                    f"run{run_idx}_"
                    f"prompt{prompt_idx}_"
                    f"history{use_history}reasoning{CONFIG['reasoning']}pair{CONFIG['pair_mode']}"
                )

                for acq_func in CONFIG["acq_functions"]:
                    filename = base.format(acq=acq_func)
                    history_filename = log_dir / f"new{RUN_ID}_{filename}.csv"

                    print("\n" + "=" * 64)
                    print(f"RUN_ID: {RUN_ID}")
                    print(f"Starting: model={model_type}, acq={acq_func}, "
                          f"run={run_idx}, prompt={prompt_idx}, history={use_history}")
                    print(f"History file: {history_filename}")
                    print("=" * 64 + "\n")
                    if CONFIG["pair_mode"]== True:
                        experiment = ScheduleComparisonExperiment(
                            schedules_df=df,
                            metric_columns=CONFIG["metric_columns"],
                            llm_client=client,
                            use_llm=not CONFIG["use_utility_sim"],
                            prompt_template=SchedulePromptTemplateAdapter(
                                reasoning_history=use_history,
                                utility_prompt=CONFIG["utility_prompt_text"],
                                metric_columns=CONFIG["metric_columns"],
                                reasoning = CONFIG['reasoning']
                            ),
                        )
                        results = experiment.run_experiment(
                            model_type=model_type,
                            batch_size=CONFIG["batch_size"],
                            n_batches=CONFIG["n_batches"],
                            n_champions=CONFIG["n_champions"],
                            acquisition=acq_func,
                            history_file=str(history_filename),
                            with_reasoning_history=use_history,
                            save_interval=CONFIG["save_interval"],
                        )
                    if CONFIG["pair_mode"] == False: 
                        experiment = ScheduleBatchExperiment(
                            schedules_df=df,
                            metric_columns=CONFIG["metric_columns"],
                            batch_size=CONFIG["batch_size"],
                            llm_client=client,
                            prompt_template=SchedulePromptTemplateAdapter(
                                reasoning_history=use_history,
                                utility_prompt=CONFIG["utility_prompt_text"],
                                metric_columns=CONFIG["metric_columns"],
                                reasoning = CONFIG['reasoning']
                            ),
                        )
                        results = experiment.run_experiment( 
                            n_batches=CONFIG["n_batches"],
                            
                            history_file= str(history_filename),
                            save_interval = 5, 
                        )
