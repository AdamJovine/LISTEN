local = True
util = False 
import glob
import pandas as pd
from LLM import FreeLLMPreferenceClient
from oss import LocalGroqLikeClient
from Experiment import ScheduleComparisonExperiment

from LISTEN import CustomDuelingBanditOptimizer, DuelingBanditOptimizer
from batchPref import BatchPrefLearning
from perfect import SimpleUtilityPreferenceClient  # Import the utility-based class
from p1 import SimpleB2BPreferenceClient 
from prompt import PromptTemplate
from singleOss import get_local_client

import os
from dotenv import load_dotenv
# Your custom prompt for the Registrar
registrar_prompt = """
You are an experienced University Registrar. 
"""

b2b_prompt = """
You are an experienced University Registrar. Your absolute top priority is ensuring students have as few back-to-back exams as possible. So minimize evening/morning b2b exams and other b2b exams.
"""

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "input/data.csv")
df = pd.read_csv(data_path)

# Filter out rows with NaN values in utility-relevant columns
utility_columns = ['evening/morning b2b', 'other b2b']
print(f"Original data shape: {df.shape}")
print(f"Rows with NaN in utility columns: {df[utility_columns].isna().any(axis=1).sum()}")

# Remove rows with NaN in utility columns
df = df.dropna(subset=utility_columns)

# Also remove rows with NaN in any metric columns to be safe


print(f"Filtered data shape: {df.shape}")
print(f"Removed {pd.read_csv(data_path).shape[0] - df.shape[0]} rows with NaN values")

# Define metrics (same as LLM version)
metric_columns = [
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
]
df = df.dropna(subset=metric_columns)
# Load environment and create LLM client
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

#client = FreeLLMPreferenceClient(
#    provider="groq",
#    api_key=api_key,
#    model_name="openai/gpt-oss-20b",
#    simple=False,
#    rate_limit_delay=0.1,
#    max_tokens=1024,
#)
if local : 
    #client = LocalGroqLikeClient(
    #    model_id="unsloth/gpt-oss-20b-BF16",   # works on 3090 with offload
    #    device_map="auto",
    #    # leave None to use the default offload for 24GB cards:
        # max_memory={0: "22GiB", "cpu": "50GiB"},
    #)
    client = get_local_client("unsloth/gpt-oss-20b-BF16")

    # Then adapt your existing client to use this "Groq-like" object:
    #class FreeLLMPreferenceClientLocal(FreeLLMPreferenceClient):
    #    def __init__(self, *args, **kwargs):
    #        # ignore api_key/provider, keep your existing knobs
    #        kwargs["provider"] = "groq"  # keep parent happy
    #        super().__init__(*args, **kwargs)
    #        self._groq = _local               # <-- swap in the local client
    #        self.model_name = "unsloth/gpt-oss-20b-BF16"  # default model
    #
    # now the rest of your code is unchanged
    #client = FreeLLMPreferenceClientLocal(simple=True)
if util : 
    client = SimpleB2BPreferenceClient()

# Configuration
#pref_modes = [ 'LLM']  # New loop for preference mode
hists = [True]  # [True, False]
prompts = [b2b_prompt, registrar_prompt]  # Your prompt templates
model_types = ["logistic"]  # ["logistic", "gp"]
acquisition_functions = ["eubo"]  # ["eubo", "ucb", "thompson", "info_gain"]

# Experimental parameters
BATCH_SIZE = 5
N_BATCHES = 50
N_CHAMPIONS = 5
UTIL_MODE = True  # True for utility simulation, False for LLM
LOCAL = True  # Your local flag

# Main experimental loops
for model_type in model_types:
    for run_idx in range(1):  # Number of runs
        for prompt_idx in [0]:  # Which prompts to use
            for use_history in hists:
                for acq_func in acquisition_functions:
                    
                    # Build history filename
                    history_filename = (
                        f"dueling_bandit_history_"
                        f"util{UTIL_MODE}_loc{LOCAL}_"
                        f"{model_type}_"
                        f"acq{acq_func}_"
                        f"run{run_idx}_"
                        f"prompt{prompt_idx}_"
                        f"history{use_history}.csv"
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"Starting run: model={model_type}, acq={acq_func}, "
                          f"run={run_idx}, prompt={prompt_idx}, history={use_history}")
                    print(f"History file: {history_filename}")
                    print(f"{'='*60}\n")
                    
                    # Create experiment runner
                    experiment = ScheduleComparisonExperiment(
                        schedules_df=df,  # Your dataframe
                        metric_columns=metric_columns,  # Your metric columns
                        llm_client= client,  # Your LLM client
                        use_llm=not UTIL_MODE,
                        prompt_template=PromptTemplate(reasoning_history = use_history , utility_prompt = prompts[prompt_idx], metric_columns = metric_columns)
                    )
                    
                    # Run experiment
                    results = experiment.run_experiment(
                        model_type=model_type,
                        batch_size=BATCH_SIZE,
                        n_batches=N_BATCHES,
                        n_champions=N_CHAMPIONS,
                        acquisition=acq_func,
                        history_file=history_filename,
                        with_reasoning_history=use_history,
                        save_interval=5  # Save every 5 batches
                    )
                    
                    # Process results
                    if 'final_ranking' in results:
                        print(f"\nExperiment completed successfully!")
                        print(f"Top schedule: {results['top_10_schedules'][0]}")
                        print(f"Total comparisons: {results['total_comparisons']}")
                    else:
                        print(f"\nExperiment failed: {results.get('error', 'Unknown error')}")
