import glob
import pandas as pd
from LLM import FreeLLMPreferenceClient
from batchPref import BatchPrefLearning
from perfect import BatchPrefLearningUtilityBased  # Import the utility-based class
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

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.csv")
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

client = FreeLLMPreferenceClient(
    provider="groq",
    api_key=api_key,
    model_name="openai/gpt-oss-20b",
    simple=False,
    rate_limit_delay=0.1,
    max_tokens=1024,
)

# Configuration
hists = [True]#[True,False]
prompts = [b2b_prompt,registrar_prompt]
pref_modes = [ 'util']  # New loop for preference mode

# Main nested loops
for pref_mode in pref_modes:
    for m in ["logistic"]:  # "gp" or 'logistic'
        for i in range(5):
            for prompt_idx in [0]:  # range(2) to use both prompts
                for hist in hists:
                    
                    # Build history filename
                    history_filename = (
                        f"perfbig_batch_pref_history_"
                        f"{pref_mode}_"  # Add preference mode to filename
                        f"{m}_"
                        f"run{i}_"
                        f"prompt{prompt_idx}_"
                        f"history{hist}.csv"
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"Starting run: {pref_mode} mode, model={m}, run={i}, prompt={prompt_idx}, history={hist}")
                    print(f"History file: {history_filename}")
                    print(f"{'='*60}\n")
                    
                    if pref_mode == 'LLM':
                        # Use LLM-based preference learning
                        bp = BatchPrefLearning(
                            schedules_df=df,
                            llm_client=client,
                            metric_columns=metric_columns,
                            history_file=history_filename,
                            m_samples=1,  # one LLM "voter" per pair
                            batch_size=5,
                            model_type=m,
                            reasoning_history=hist,
                        )
                        
                        # Run with prompt for LLM mode
                        final_rankings = bp.run(
                            n_batches=50, 
                            prompt_init=prompts[prompt_idx], 
                            save_history=True
                        )
                        
                    else:  # pref_mode == 'util'
                        # Use utility-based preference learning
                        bp = BatchPrefLearningUtilityBased(
                            schedules_df=df,
                            llm_client=None,  # No LLM client needed
                            metric_columns=metric_columns,
                            history_file=history_filename,
                            m_samples=1,  # Keep consistent with LLM version
                            batch_size=5,
                            model_type=m,
                            reasoning_history=hist,  # Will be ignored but kept for consistency
                        )
                        
                        # Run without prompt for utility mode (prompt will be ignored)
                        final_rankings = bp.run(
                            n_batches=50,
                            prompt_init=None,  # No prompt needed for utility-based
                            save_history=True
                        )
                    
                    # Inspect top schedules
                    print(f"\nTop 5 schedules for {pref_mode} mode:")
                    print(final_rankings.head())
                    
                    # Optional: Save final rankings with mode info
                    rankings_filename = history_filename.replace('.csv', '_final_rankings.csv')
                    final_rankings['pref_mode'] = pref_mode
                    final_rankings['model_type'] = m
                    final_rankings['run_id'] = i
                    final_rankings['prompt_idx'] = prompt_idx
                    final_rankings['history_enabled'] = hist
                    final_rankings.to_csv(rankings_filename, index=False)
                    print(f"Saved final rankings to: {rankings_filename}")

print("\n" + "="*60)
print("ALL RUNS COMPLETE")
print("="*60)