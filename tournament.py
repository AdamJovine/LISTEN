"""
Tournament Selection for Schedule Optimization using LLM Oracle
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import random
import json
from remoteOss import get_local_client


class TournamentSelection:
    """
    Tournament selection using LLM to evaluate schedules based on key metrics.
    """
    
    # Key metrics to consider for schedule evaluation
    EVAL_METRICS = [
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
    
    def __init__(
        self,
        csv_path: str = "input/data.csv",
        batch_size: int = 50,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the tournament selection.
        
        Args:
            csv_path: Path to the CSV file containing schedule data
            batch_size: Number of schedules to show in each batch (default 50)
            seed: Random seed for reproducibility
            verbose: Whether to print progress messages
        """
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load the data
        self.df = self._load_data()
        
        # Get LLM client
        self.llm_client = get_local_client()
        
        if self.verbose:
            print(f"Loaded {len(self.df)} valid schedules from {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate the CSV data."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Check for required columns
        missing_cols = set(self.EVAL_METRICS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out rows with NaN values in evaluation metrics
        initial_count = len(df)
        df = df.dropna(subset=self.EVAL_METRICS)
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0 and self.verbose:
            print(f"Filtered out {filtered_count} rows with NaN values in evaluation metrics")
        
        if len(df) == 0:
            raise ValueError("No valid rows remaining after filtering NaN values")
        
        # Reset index to ensure continuous indexing
        df = df.reset_index(drop=True)
        
        # Add index column if not present
        if df.index.name != 'schedule_id':
            df.index.name = 'schedule_id'
        
        return df
    
    def _format_schedule_for_llm(self, schedule_id: int, row: pd.Series) -> str:
        """Format a single schedule's metrics for LLM presentation."""
        lines = [f"Schedule {schedule_id}:"]
        for metric in self.EVAL_METRICS:
            value = row[metric]
            # Format the value appropriately
            if isinstance(value, float):
                formatted_value = f"{value:.1f}" if value != int(value) else str(int(value))
            else:
                formatted_value = str(value)
            
            lines.append(f"  - {metric}: {formatted_value}")
        return "\n".join(lines)
    
    def _create_batch_prompt(self, batch_df: pd.DataFrame) -> str:
        """Create a prompt for evaluating a batch of schedules."""
        prompt_parts = [
            "You are an expert at evaluating exam schedules. Your goal is to identify schedules that minimize student conflicts and stress.",
            "",
            "Key metrics to consider (lower is better for all):",
            "- conflicts: Direct exam time conflicts",
            "- quints/quads: Students with 5 or 4 exams in a short period",
            "- four in five slots / three in four slots: Exam clustering",
            "- triple in 24h / triple in same day: Three exams in quick succession",
            "- evening/morning b2b / other b2b: Back-to-back exams",
            "- two in three slots: Two exams with minimal gap",
            "",
            "Please evaluate the following schedules and select the SINGLE BEST one that minimizes student stress and conflicts:",
            ""
        ]
        
        # Add each schedule in randomized order
        schedule_ids = list(batch_df.index)
        random.shuffle(schedule_ids)
        
        for i, sid in enumerate(schedule_ids, 1):
            row = batch_df.loc[sid]
            prompt_parts.append(self._format_schedule_for_llm(sid, row))
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Based on the metrics above, which schedule is the BEST overall?",
            "Respond with ONLY the schedule number (e.g., '42')."
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_final_prompt(self, finalist_df: pd.DataFrame) -> str:
        """Create a prompt for final selection of top 5 schedules."""
        prompt_parts = [
            "You are an expert at evaluating exam schedules. These are the finalist schedules from preliminary rounds.",
            "",
            "Key metrics to consider (lower is better for all):",
            "- conflicts: Direct exam time conflicts",
            "- quints/quads: Students with 5 or 4 exams in a short period",
            "- four in five slots / three in four slots: Exam clustering",
            "- triple in 24h / triple in same day: Three exams in quick succession",
            "- evening/morning b2b / other b2b: Back-to-back exams",
            "- two in three slots: Two exams with minimal gap",
            "",
            "Please evaluate these finalist schedules and select the TOP 5 that best minimize student stress and conflicts:",
            ""
        ]
        
        # Add finalists in randomized order
        schedule_ids = list(finalist_df.index)
        random.shuffle(schedule_ids)
        
        for i, sid in enumerate(schedule_ids, 1):
            row = finalist_df.loc[sid]
            prompt_parts.append(self._format_schedule_for_llm(sid, row))
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Select the TOP 5 schedules that are best overall.",
            "Respond with ONLY the 5 schedule numbers, one per line, in order from best to 5th best.",
            "Example format:",
            "42",
            "17",
            "8",
            "23",
            "31"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_single_selection(self, response: str) -> Optional[int]:
        """Parse a single schedule selection from LLM response."""
        try:
            # Try to extract the first number from the response
            import re
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                return int(numbers[0])
        except:
            pass
        return None
    
    def _parse_multiple_selections(self, response: str, max_count: int = 5) -> List[int]:
        """Parse multiple schedule selections from LLM response."""
        selections = []
        try:
            import re
            # Find all numbers in the response
            numbers = re.findall(r'\d+', response.strip())
            for num_str in numbers[:max_count]:
                selections.append(int(num_str))
        except:
            pass
        return selections
    
    def run_tournament(self) -> Tuple[List[int], pd.DataFrame]:
        """
        Run the tournament selection process.
        
        Returns:
            Tuple of (list of top 5 schedule IDs, DataFrame with their metrics)
        """
        if self.verbose:
            print(f"\nStarting tournament with {len(self.df)} schedules")
            print(f"Batch size: {self.batch_size}")
        
        # Phase 1: Process schedules in batches
        finalists = []
        n_batches = (len(self.df) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.df))
            batch_df = self.df.iloc[start_idx:end_idx]
            
            if self.verbose:
                print(f"\nProcessing batch {batch_idx + 1}/{n_batches} (schedules {start_idx}-{end_idx-1})")
            
            # Create prompt and get selection
            prompt = self._create_batch_prompt(batch_df)
            
            # Call LLM (using the call_oracle method with dummy schedules since we're repurposing it)
            _, raw_response = self.llm_client.call_oracle(
                prompt=prompt,
                sched_a="dummy_a",  # Not used in our prompt
                sched_b="dummy_b",  # Not used in our prompt
                temperature=0.1,
                max_new_tokens=100
            )
            
            # Parse the selection
            selected_id = self._parse_single_selection(raw_response)
            
            if selected_id is not None and selected_id in batch_df.index:
                finalists.append(selected_id)
                if self.verbose:
                    print(f"  Selected: Schedule {selected_id}")
            else:
                # Fallback: select the schedule with lowest total penalty
                penalty_sum = batch_df[self.EVAL_METRICS].sum(axis=1)
                best_id = penalty_sum.idxmin()
                finalists.append(best_id)
                if self.verbose:
                    print(f"  Fallback selection: Schedule {best_id} (parsing failed)")
        
        if self.verbose:
            print(f"\n{len(finalists)} finalists selected from batches")
        
        # Phase 2: Final selection from finalists
        if len(finalists) <= 5:
            # If we have 5 or fewer finalists, return them all
            top_5 = finalists
        else:
            finalist_df = self.df.loc[finalists]
            
            if self.verbose:
                print(f"\nRunning final selection round with {len(finalists)} finalists...")
            
            # Create final prompt
            final_prompt = self._create_final_prompt(finalist_df)
            
            # Call LLM for final selection
            _, raw_response = self.llm_client.call_oracle(
                prompt=final_prompt,
                sched_a="dummy_a",
                sched_b="dummy_b",
                temperature=0.1,
                max_new_tokens=200
            )
            
            # Parse top 5 selections
            top_5 = self._parse_multiple_selections(raw_response, max_count=5)
            
            # Validate selections
            top_5 = [sid for sid in top_5 if sid in finalists]
            
            # If parsing failed or we didn't get enough, fill with best by total penalty
            if len(top_5) < 5:
                penalty_sum = finalist_df[self.EVAL_METRICS].sum(axis=1)
                sorted_ids = penalty_sum.sort_values().index.tolist()
                for sid in sorted_ids:
                    if sid not in top_5:
                        top_5.append(sid)
                    if len(top_5) >= 5:
                        break
            
            top_5 = top_5[:5]
        
        # Get the final results DataFrame
        results_df = self.df.loc[top_5][self.EVAL_METRICS]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("TOURNAMENT COMPLETE - TOP 5 SCHEDULES:")
            print('='*60)
            for rank, sid in enumerate(top_5, 1):
                print(f"\nRank {rank}: Schedule {sid}")
                for metric in self.EVAL_METRICS:
                    value = self.df.loc[sid, metric]
                    if isinstance(value, float):
                        formatted_value = f"{value:.1f}" if value != int(value) else str(int(value))
                    else:
                        formatted_value = str(value)
                    print(f"  {metric}: {formatted_value}")
        
        return top_5, results_df
    
    def get_schedule_details(self, schedule_id: int) -> pd.Series:
        """Get full details for a specific schedule."""
        if schedule_id not in self.df.index:
            raise ValueError(f"Schedule {schedule_id} not found")
        return self.df.loc[schedule_id]


# Example usage
if __name__ == "__main__":
    # Initialize the tournament selection
    tournament = TournamentSelection(
        csv_path="input/data.csv",
        batch_size=50,
        seed=42,
        verbose=True
    )
    
    # Run the tournament
    top_5_ids, top_5_metrics = tournament.run_tournament()
    
    print("\n" + "="*60)
    print("Final Results Summary:")
    print("="*60)
    print(f"Top 5 Schedule IDs: {top_5_ids}")
    print("\nMetrics DataFrame:")
    print(top_5_metrics)
    
    # Optionally save results
    top_5_metrics.to_csv("tournament_results.csv")
    print("\nResults saved to tournament_results.csv")