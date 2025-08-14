import pandas as pd
import numpy as np
import concurrent.futures
import time
import random
from collections import defaultdict, Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional, Tuple, Set
import os
from scipy.stats import norm
from scipy.special import expit
from batchPref import BatchPrefLearning

class BatchPrefLearningUtilityBased(BatchPrefLearning):
    """
    Extended BatchPrefLearning that uses utility-based decisions instead of LLM calls.
    Utility function: -(evening/morning b2b + other b2b)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute utilities for all schedules based on the specified formula
        self.schedule_utilities = self._compute_utilities()
        
        # Create a mapping from schedule index to utility
        # This ensures we can look up utilities correctly regardless of indexing
        self.idx_to_utility = {}
        for i in range(len(self.df)):
            self.idx_to_utility[i] = self.schedule_utilities[i]

    def _compute_utilities(self) -> np.ndarray:
        """
        Compute utilities for all schedules using the formula:
        utility = -(evening/morning b2b + other b2b)
        """
        utilities = -(
            self.df['evening/morning b2b'].astype(float) +
            self.df['other b2b'].astype(float)
        )
        return utilities.to_numpy()

    def _make_utility_based_comparison(
        self, idx_a: int, idx_b: int
    ) -> Tuple[str, float, float]:
        """
        Compare two schedules based on their utilities.
        Returns: (winner, utility_a, utility_b)
        """
        # Use the mapping to ensure correct utility retrieval
        utility_a = self.idx_to_utility[idx_a]
        utility_b = self.idx_to_utility[idx_b]

        if utility_a > utility_b:
            winner = "A"
        elif utility_b > utility_a:
            winner = "B"
        else:
            # In case of tie, randomly pick winner
            winner = "A" if random.random() > 0.5 else "B"

        return winner, utility_a, utility_b

    def _make_single_comparison(
        self, sched_a: Dict, sched_b: Dict, prompt_init: Optional[str], sample_id: int
    ):
        """
        Override to use utility-based comparison instead of LLM.
        """
        start_time = time.time()

        # Get indices from the schedule dictionaries
        idx_a = self._find_schedule_index(sched_a)
        idx_b = self._find_schedule_index(sched_b)

        winner, utility_a, utility_b = self._make_utility_based_comparison(idx_a, idx_b)

        # Create a reason based on utilities
        reason = f"Schedule {winner} has higher utility ({utility_a if winner == 'A' else utility_b:.3f}) " \
                f"compared to Schedule {'B' if winner == 'A' else 'A'} " \
                f"({utility_b if winner == 'A' else utility_a:.3f})"

        # Store in voter reflections for consistency
        self.voter_reflections[sample_id].append({
            "sched_a": sched_a,
            "sched_b": sched_b,
            "winner": winner,
            "reason": reason
        })

        elapsed = time.time() - start_time
        print(f"    Voter {sample_id}: {winner} (utility-based, {elapsed:.3f}s)")

        # Return format consistent with original
        return sample_id, (winner, reason, None), f"Utility-based comparison: {reason}"

    def _find_schedule_index(self, sched_dict: Dict) -> int:
        """
        Find the index of a schedule given its metric dictionary.
        """
        # Convert schedule dict to array for comparison
        sched_values = np.array([sched_dict[col] for col in self.metrics])

        # Find matching row in feature matrix
        for idx in range(len(self.feat)):
            if np.allclose(self.feat[idx], sched_values, rtol=1e-9):
                return idx

        raise ValueError(f"Could not find schedule index for {sched_dict}")

    def _make_multiple_llm_votes(
        self, idx_a: int, idx_b: int, n_samples: int, prompt_init: Optional[str] = None
    ):
        """
        Override to use utility-based voting instead of LLM calls.
        All voters will agree since utility is deterministic.
        """
        sched_a = dict(zip(self.metrics, self.feat[idx_a]))
        sched_b = dict(zip(self.metrics, self.feat[idx_b]))
        print('sched a ',sched_a )
        sched_b = self.feat[idx_b]
        print('sche b' , sched_b)
        # evening/morning b2b'].astype(float) + self.df['other b2b' 
        utility_a = -sched_a['evening/morning b2b'] - sched_b['other b2b']
        utility_b = -sched_b['evening/morning b2b'] - sched_b['other b2b']
        #winner =  #utility_a, utility_b = #self._make_utility_based_comparison(idx_a, idx_b)
        winner = 'A'
        if utility_a > utility_b:
            winner = "A"
        elif utility_b > utility_a:
            winner = "B"
        # Since utility is deterministic, all "voters" would pick the same winner
        votes = {"A": 0, "B": 0}
        votes[winner] = n_samples

        # Create consistent responses for all voters
        reason = f"Utility: A={utility_a:.3f}, B={utility_b:.3f}"
        responses = [(winner, reason, None) for _ in range(n_samples)]

        # Create schedule dictionaries for consistency


        # Update voter reflections
        for voter_id in range(n_samples):
            self.voter_reflections[voter_id].append({
                "sched_a": sched_a,
                "sched_b": sched_b,
                "winner": winner,
                "reason": reason
            })

        # Calculate stats (will have 0 entropy and max confidence since all voters agree)
        total_votes = n_samples
        vote_ratio_a = votes["A"] / total_votes
        vote_ratio_b = votes["B"] / total_votes

        # Entropy is 0 when all votes go to one option
        entropy = 0.0

        # Maximum confidence since all voters agree
        confidence = 1.0
        vote_margin = 1.0

        stats = {
            "total_votes": total_votes,
            "vote_ratio_a": vote_ratio_a,
            "vote_ratio_b": vote_ratio_b,
            "entropy": entropy,
            "vote_margin": vote_margin,
            "confidence": confidence,
            "winner": winner,
            "utility_a": float(utility_a),
            "utility_b": float(utility_b),
            "utility_diff": float(abs(utility_a - utility_b))
        }

        full_prompt = f"Utility-based comparison: A={utility_a:.3f}, B={utility_b:.3f}, Winner={winner}"
        reflection_summary = None

        print(f"    Result: {votes['A']}-{votes['B']} (utility diff: {abs(utility_a - utility_b):.3f})")

        return votes, responses, reflection_summary, full_prompt, stats

    def _collect_batch_comparisons(
        self, pairs: List[Tuple[int, int]], prompt_init: Optional[str] = None
    ) -> List[Dict]:
        """
        Override to handle utility-based comparisons more efficiently.
        """
        print(f"\nCollecting utility-based comparisons for batch of {len(pairs)} pairs...")
        batch_results: List[Dict] = []
        batch_winners: List[int] = []

        for k, (idx_a, idx_b) in enumerate(pairs):
            print(f"\n  Pair {k+1}/{len(pairs)}: {idx_a} vs {idx_b}")

            # Utility-based comparison
            #winner, utility_a, utility_b = self._make_utility_based_comparison(idx_a, idx_b)
            sched_a = dict(zip(self.metrics, self.feat[idx_a]))
            sched_b = dict(zip(self.metrics, self.feat[idx_b]))
            print('sched a ',sched_a )
            print('sche b' , sched_b)
            # evening/morning b2b'].astype(float) + self.df['other b2b' 
            utility_a = -sched_a['evening/morning b2b'] - sched_b['other b2b']
            utility_b = -sched_b['evening/morning b2b'] - sched_b['other b2b']
            #winner =  #utility_a, utility_b = #self._make_utility_based_comparison(idx_a, idx_b)
            winner = 'A'
            if utility_a > utility_b:
                winner = "A"
            elif utility_b > utility_a:
                winner = "B"
            # Create consistent vote results (all voters agree in utility-based mode)
            votes = {"A": 0, "B": 0}
            votes[winner] = self.m_samples

            champion = idx_a if winner == "A" else idx_b
            batch_winners.append(champion)

            # Create responses for consistency - FIXED TYPO HERE
            reason = f"Utility: A={utility_a:.3f}, B={utility_b:.3f}"
            responses = [(winner, reason, None) for _ in range(self.m_samples)]

            # Stats with perfect agreement
            stats = {
                "batch_num": getattr(self, 'current_batch_num', k),
                "total_comparison_num": len(self.history) + 1,
                "winner": winner,
                "total_votes": self.m_samples,
                "vote_ratio_a": 1.0 if winner == "A" else 0.0,
                "vote_ratio_b": 1.0 if winner == "B" else 0.0,
                "entropy": 0.0,
                "confidence": 1.0,
                "vote_margin": 1.0,
                "utility_a": float(utility_a),
                "utility_b": float(utility_b),
                "utility_diff": float(abs(utility_a - utility_b)),
                "error": None
            }

            result = {
                "idx_a": idx_a,
                "idx_b": idx_b,
                "champion_idx": champion,
                "votes": votes,
                "responses": responses,
                "group_reflection": None,
                "full_prompt": f"Utility comparison: {reason}",
                **stats
            }

            batch_results.append(result)
            print(f"    Utilities: A={utility_a:.3f}, B={utility_b:.3f}")
            print(f"    Winner: {champion} (Schedule {winner})")

            # Update cumulative votes
            self._update_cumulative_votes(idx_a, idx_b, votes)
            self.compared_pairs.add((min(idx_a, idx_b), max(idx_a, idx_b)))

        self.previous_winners = list(set(batch_winners))
        return batch_results

    def run(
        self,
        n_batches: int = 20,
        prompt_init: Optional[str] = None,
        save_history: bool = True,
    ) -> pd.DataFrame:
        """
        Override run to track batch numbers properly.
        """
        print(f"Starting UTILITY-BASED batch preference learning (model={self.model_type})")
        print(f"Utility function: -(evening/morning b2b + other b2b)")
        print(f"Batches: {n_batches}, Batch size: {self.batch_size}, Voters: {self.m_samples}")
        print(f"Note: All voters will agree since utility is deterministic\n")

        for batch_num in range(1, n_batches + 1):
            self.current_batch_num = batch_num  # Track current batch
            print(f"\n{'='*60}\nBATCH {batch_num}/{n_batches}\n{'='*60}")

            if batch_num == 1:
                pairs = self._select_initial_batch()
                print(f"Initial batch: selected {len(pairs)} random pairs")
            else:
                pairs = self._select_batch_with_model()
                print(f"Model-guided batch: selected {len(pairs)} pairs using {self.model_type}")

            batch_results = self._collect_batch_comparisons(pairs, prompt_init)
            self._update_model_batch(batch_results)

            batch_summary = {
                "batch_num": batch_num,
                "n_comparisons": len(batch_results),
                "total_comparisons_so_far": len(self.X_delta),
                "pairs": pairs,
                "results": batch_results,
            }
            self.batch_history.append(batch_summary)

            for i, r in enumerate(batch_results):
                r["batch_num"] = batch_num
                r["comparison_num_in_batch"] = i + 1
                r["total_comparison_num"] = len(self.history) + 1
                self.history.append(r)

            print(f"\nCurrent top 5 schedules after batch {batch_num}:")
            current_rankings = self._get_current_rankings()

            # Add true utilities to rankings for reference
            rankings_with_utility = current_rankings.copy()
            rankings_with_utility['true_utility'] = [
                self.idx_to_utility[idx] for idx in rankings_with_utility['schedule_idx'].values
            ]

            print(rankings_with_utility.head()[
                ["schedule_idx", "model_score", "true_utility", "win_rate", "total_votes"]
            ])

        final_rankings = self._get_current_rankings()
        final_rankings['true_utility'] = [
            self.idx_to_utility[idx] for idx in final_rankings['schedule_idx'].values
        ]

        if save_history:
            self._save_history()
            self._save_batch_summary()

        print(f"\nCompleted {n_batches} batches with {len(self.history)} total comparisons")
        print("\nNote: All comparisons were made using the utility function:")
        print("utility = -(evening/morning b2b + other b2b)")

        return final_rankings