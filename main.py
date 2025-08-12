import glob
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from LLM import FreeLLMPreferenceClient
from AcqStrategy import AcquisitionStrategy
from batchPref import BatchPrefLearning
import os
from dotenv import load_dotenv

# Your custom prompt for the Registrar
registrar_prompt = """
You are an experienced University Registrar. Your absolute top priority is ensuring no student has a simultaneous conflict.
After that, your next most critical goal is to minimize the number of students facing three exams in a 24-hour period, 
as this causes the most stress. 
"""
b2b_prompt = """
You are an experienced University Registrar. Your absolute top priority is ensuring students have as few back-to-back exams as possible. So minimize evening/morning b2b exams and other b2b exams.
"""

paths = glob.glob("../BOScheduling/results/sp25/metrics/*.csv")
if not paths:
    print("No schedule files found.")
    exit(1)

df = pd.read_csv("data.csv")

print(f"Loaded {len(df)} schedules from {len(paths)} files")

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
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = FreeLLMPreferenceClient(
    provider="groq",
    api_key=api_key,
    model_name="openai/gpt-oss-120b",
    simple=False,
    rate_limit_delay=0.1,
    max_tokens=1024,
)
hist = False
prompts = [b2b_prompt]
for m in ["logistic"]:  # "gp_eubo",
    for i in range(50):
        for prompt in range(2):
            # Instantiate with Gaussian Process, 1 sample per comparison
            bp = BatchPrefLearning(
                schedules_df=df,  # your DataFrame of candidate schedules
                llm_client=client,  # your wrapped LLM client
                metric_columns=metric_columns,
                history_file="newbig_batch_pref_history"
                + m
                + str(i)
                + "prompt"
                + str(prompt)
                + "history"
                + str(hist)
                + ".csv",  # where to save detailed history
                m_samples=1,  # one LLM “voter” per pair
                batch_size=5,  # or whatever batch size you prefer
                model_type=m,  # use Gaussian Process regression
                reasoning_history=hist,
            )
            # Run for, say, 10 batches, seeding in your prompt each time
            final_rankings = bp.run(
                n_batches=50, prompt_init=b2b_prompt, save_history=True
            )
            # Inspect top schedules
            print(final_rankings.head())
