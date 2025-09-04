import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import json
import csv
from datetime import datetime
from LISTEN import CustomDuelingBanditOptimizer
from oracle import Oracle
from promptTemplate import PromptTemplateInterface

from prompt import ComparisonPromptTemplate, SchedulePromptTemplateAdapter

class ScheduleComparisonExperiment:
    """
    Wrapper to run schedule comparison experiments using DuelingBanditOptimizer
    with support for history tracking and multiple experimental conditions.
    """
    
    def __init__(
        self,
        schedules_df: pd.DataFrame,
        metric_columns: List[str],
        llm_client: Oracle,
        use_llm: bool = False,
        prompt_template:ComparisonPromptTemplate = SchedulePromptTemplateAdapter(), 
    ):
        """
        Args:
            schedules_df: DataFrame containing schedules
            metric_columns: Columns to use as features
            llm_client: LLM client for preference collection (optional)
            use_llm: Whether to use LLM for preferences
            prompt_template: Template for LLM prompts
        """
        self.schedules_df = schedules_df
        self.metric_columns = metric_columns
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.prompt_template = prompt_template
        
        # Extract features from dataframe
        self.features = schedules_df[metric_columns].values
        self.n_schedules = len(schedules_df)
        self.schedule_indices = list(range(self.n_schedules))
        
        # Normalize features
        #self.features = (self.features - self.features.mean(axis=0)) / (self.features.std(axis=0) + 1e-8)
        
    def collect_llm_preference(self, idx_a: int, idx_b: int, with_history: bool = False) -> int:
        """
        Collect preference from LLM for two schedules.
        
        Returns:
            Index of winning schedule
        """
        #if not self.use_llm or self.llm_client is None:
        #    # Fallback to utility-based simulation
        #    return self.simulate_utility_preference(idx_a, idx_b)
        
        schedule_a = self.schedules_df.iloc[idx_a]
        schedule_b = self.schedules_df.iloc[idx_b]
        sched_a = schedule_a.to_dict()
        sched_b = schedule_b.to_dict()
        # Format prompt with schedules
        prompt = self.prompt_template.format([sched_a,sched_b])
        print('prompt ' , prompt)
        # Call LLM (implement based on your LLM client)
        desicion,reasoning = self.llm_client.call_oracle(prompt, sched_a , sched_b )
        print('reasoning, ' , reasoning )
        print('decision ' , desicion)
        self.prompt_template.add_to_history(sched_a, desicion, reasoning)
        self.prompt_template.add_to_history(sched_b, desicion, reasoning)
        if "A" in desicion :
            return idx_a 
        if "B" in desicion :
            return idx_b
        return -1 

    
    #def simulate_utility_preference(self, idx_a: int, idx_b: int) -> int:
    #    """
    #    Simulate preference based on ground truth utility.
    #    Used when not using LLM.
    #    """
    #    # Define ground truth weights for simulation
    #    n_features = self.features.shape[1]
    #    if not hasattr(self, 'true_weights'):
    #        np.random.seed(123)  # Consistent ground truth
    #        self.true_weights = np.random.randn(n_features)
    #        self.true_weights = self.true_weights / np.linalg.norm(self.true_weights)
    #    # Compute utilities
    #    util_a = self.features[idx_a] @ self.true_weights
    #    util_b = self.features[idx_b] @ self.true_weights
    #    
    #    # Probabilistic choice with logistic noise
    #    prob_a = 1 / (1 + np.exp(-2 * (util_a - util_b)))  # Temperature = 0.5
    #    
    #    return idx_a if np.random.random() < prob_a else idx_b
    
    def run_experiment(
        self,
        model_type: str,
        batch_size: int,
        n_batches: int,
        n_champions: int,
        acquisition: str,
        history_file: str,
        with_reasoning_history: bool = False,
        save_interval: int = 5
    ) -> Dict:
        """
        Run a single experiment with DuelingBanditOptimizer.
        
        Args:
            model_type: Type of model ('logistic' or 'gp')
            batch_size: Number of comparisons per batch
            n_batches: Number of batches to run
            n_champions: Number of champions to maintain
            acquisition: Acquisition function type
            history_file: File to save history
            with_reasoning_history: Whether to include reasoning in LLM prompts
            save_interval: Save history every N batches
            
        Returns:
            Results dictionary
        """
        print(f"Starting experiment: {model_type} model, {acquisition} acquisition")
        print(f"Batch size: {batch_size}, N batches: {n_batches}, N champions: {n_champions}")
        
        # Initialize history tracking
        history = {
            'metadata': {
                'model_type': model_type,
                'batch_size': batch_size,
                'n_batches': n_batches,
                'n_champions': n_champions,
                'acquisition': acquisition,
                'use_llm': self.use_llm,
                'with_reasoning': with_reasoning_history,
                'start_time': datetime.now().isoformat()
            },
            'comparisons': [],
            'batch_results': [],
            'utilities': []
        }
        
        # Create custom optimizer that uses our preference collection
        optimizer = CustomDuelingBanditOptimizer(
            all_options=self.schedule_indices,
            features=self.features,
            batch_size=batch_size,
            client = self.llm_client , 
            n_champions=n_champions,
            acquisition=acquisition,
            C=1.0 if model_type == 'logistic' else 0.1,
            preference_collector=lambda a, b: self.collect_llm_preference(
                a, b, with_history=with_reasoning_history
            )
        )
        
        # Run optimization
        for batch_idx in range(n_batches):
            print(f"\n--- Batch {batch_idx + 1}/{n_batches} ---")
            
            # Get next batch of comparisons
            pairs = optimizer.select_next_batch()
            
            if not pairs:
                print("No more pairs to compare, stopping early")
                break
            
            # Collect preferences
            batch_comparisons = []
            for idx_a, idx_b in pairs:
                winner= self.collect_llm_preference(
                    idx_a, idx_b, 
                    with_history=with_reasoning_history
                )
                
                comparison = {
                    'batch': batch_idx,
                    'option_a': idx_a,
                    'option_b': idx_b,
                    'sched_a' : self.schedules_df.iloc[idx_a].to_dict(), 
                    'sched_b' : self.schedules_df.iloc[idx_b].to_dict(), 
                    'winner': winner,
                    'timestamp': datetime.now().isoformat()
                }
                print('comparison' , comparison)
                
                batch_comparisons.append(comparison)
                history['comparisons'].append(comparison)
            
            # Update model with new preferences
            optimizer.update_with_comparisons(batch_comparisons)
            
            # Get current state
            if optimizer.model.ready():
                utilities = optimizer.model.posterior_mean_util(self.features)
                uncertainties = optimizer.model.posterior_std_util(self.features)
                
                # Top schedules
                top_5 = optimizer.current_champions#np.argsort(utilities)[-5:][::-1]
                
                batch_result = {
                    'batch': batch_idx,
                    'top_5_indices': str(top_5),
                    'top_5_utilities': utilities[top_5].tolist(),
                    'top_5_uncertainties': uncertainties[top_5].tolist(),
                    'current_champions': optimizer.current_champions
                }
                
                history['batch_results'].append(batch_result)
                history['utilities'].append(utilities.tolist())
                
                print(f"Current top 5 schedules: {top_5}")
                print(f"Utilities: {utilities[top_5].round(3)}")
                
            
            # Save periodically
            if (batch_idx + 1) % save_interval == 0:
                self.save_history(history, history_file)
        
        # Final save
        history['metadata']['end_time'] = datetime.now().isoformat()
        self.save_history(history, history_file)
        
        # Prepare final results
        if optimizer.model.ready():
            final_utilities = optimizer.model.posterior_mean_util(self.features)
            final_ranking = np.argsort(final_utilities)[::-1]
            
            results = {
                'final_ranking': final_ranking.tolist(),
                'final_utilities': final_utilities[final_ranking].tolist(),
                'top_10_schedules': final_ranking[:10].tolist(),
                'total_comparisons': len(history['comparisons']),
                'history_file': history_file
            }
            
            print(f"\n--- Final Results ---")
            print(f"Top 10 schedules: {final_ranking[:10]}")
            print(f"Total comparisons: {len(history['comparisons'])}")
            
            return results
        
        return {'error': 'Model did not converge', 'history_file': history_file}
    
    def save_history(self, history: Dict, filename: str):
        """Save history to CSV and JSON files."""
        # Save detailed JSON
        json_file = filename.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save comparisons to CSV for compatibility
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'batch', 'option_a', 'option_b', 'sched_a','sched_b','winner', 'timestamp'
            ])
            writer.writeheader()
            writer.writerows(history['comparisons'])
        
        print(f"History saved to {filename} and {json_file}")

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import csv
class ScheduleBatchExperiment:
    """
    Schedule comparison experiment using batch preference elicitation.
    Shows batches of schedules and asks for the best one, carrying winner forward.
    """
    
    def __init__(
        self,
        schedules_df: pd.DataFrame,
        metric_columns: List[str],
        llm_client,  # Oracle type
        prompt_template=None,  # BatchSchedulePromptTemplate
        batch_size: int = 4,
        n_schedules_to_explore: int = None
    ):
        """
        Args:
            schedules_df: DataFrame containing schedules
            metric_columns: Columns to use as features
            llm_client: LLM client for preference collection
            prompt_template: BatchSchedulePromptTemplate instance
            batch_size: Number of schedules to show per batch
            n_schedules_to_explore: Total schedules to explore (None = all)
        """
        self.schedules_df = schedules_df
        self.metric_columns = metric_columns
        self.llm_client = llm_client
        self.batch_size = batch_size
        
        # Initialize prompt template if not provided
        if prompt_template is None:
            from prompt import BatchSchedulePromptTemplate
            self.prompt_template = BatchSchedulePromptTemplate(
                reasoning=True,
                metric_columns=metric_columns
            )
        else:
            self.prompt_template = prompt_template
        
        # Extract features and setup
        self.features = schedules_df[metric_columns].values
        self.n_schedules = len(schedules_df)
        self.n_schedules_to_explore = n_schedules_to_explore or self.n_schedules
        
        # Track which schedules have been shown and their win counts
        self.schedule_wins = np.zeros(self.n_schedules)  # Win counts
        self.schedule_appearances = np.zeros(self.n_schedules)  # Times shown
        self.schedule_scores = np.zeros(self.n_schedules)  # Bradley-Terry scores
        
        # Track schedules that haven't been evaluated yet
        self.unshown_indices = list(range(self.n_schedules))
        np.random.shuffle(self.unshown_indices)
        
        # Track the current winner to carry forward
        self.current_winner_idx = None
        
        # History of all winners for debugging
        self.winner_history = []
        
    def collect_batch_preference(self, batch_indices: List[int]) -> Dict[str, Any]:
        """
        Collect preference from LLM for a batch of schedules.
        
        Args:
            batch_indices: Indices of schedules to compare
            
        Returns:
            Dictionary with winner index, choice letter, and reasoning
        """
        # Get schedule dictionaries
        schedules = []
        for idx in batch_indices:
            schedule = self.schedules_df.iloc[idx].to_dict()
            schedules.append(schedule)
        
        # Format prompt
        prompt = self.prompt_template.format(schedules)
        print(f"\n{'='*50}")
        print("Batch comparison prompt:")
        print(prompt)
        print(f"{'='*50}\n")
        
        # Call LLM
        response = self.llm_client.generate_response(prompt, max_new_tokens=4096)
        print(f"LLM Response:\n{response}\n")
        
        # Parse response
        choice_letter = self.prompt_template.parse_response(response)
        
        # Convert letter to index
        winner_idx_in_batch = ord(choice_letter) - ord('A')
        winner_idx = batch_indices[winner_idx_in_batch]
        
        # Extract reasoning if present
        reasoning = None
        if "FINAL:" in response:
            reasoning = response.split("FINAL:")[0].strip()
        
        # Add to history
        self.prompt_template.add_to_history(schedules, choice_letter, reasoning)
        
        return {
            'winner_idx': winner_idx,
            'choice_letter': choice_letter,
            'reasoning': reasoning,
            'batch_indices': batch_indices,
            'schedules': schedules
        }
    
    def select_next_batch(self, batch_num: int) -> List[int]:
        """
        Select next batch of schedules to compare.
        SIMPLIFIED: Always include previous winner + (batch_size - 1) new schedules.
        
        Args:
            batch_num: Current batch number (0-indexed)
            
        Returns:
            List of schedule indices for next batch
        """
        batch = []
        
        # For first batch, just pick batch_size random schedules
        if batch_num == 0 or self.current_winner_idx is None:
            # Initial batch: randomly select batch_size schedules
            if len(self.unshown_indices) >= self.batch_size:
                # Pick batch_size new schedules
                selected = self.unshown_indices[:self.batch_size]
                batch = selected.copy()
                # Remove from unshown
                for idx in selected:
                    self.unshown_indices.remove(idx)
            else:
                # Not enough unshown, use what we have + random from shown
                batch = self.unshown_indices.copy()
                self.unshown_indices = []
                
                # Fill remainder with random already-shown schedules
                shown_indices = [i for i in range(self.n_schedules) 
                               if self.schedule_appearances[i] > 0 and i not in batch]
                remaining_needed = self.batch_size - len(batch)
                if shown_indices and remaining_needed > 0:
                    additional = np.random.choice(shown_indices, 
                                                size=min(remaining_needed, len(shown_indices)),
                                                replace=False).tolist()
                    batch.extend(additional)
        else:
            # Subsequent batches: ALWAYS include the previous winner
            batch.append(self.current_winner_idx)
            print(f"  Including previous winner: {self.current_winner_idx}")
            
            # Add (batch_size - 1) new schedules
            new_schedules_needed = self.batch_size - 1
            
            # First try to get unshown schedules
            if len(self.unshown_indices) >= new_schedules_needed:
                # We have enough unshown schedules
                selected = self.unshown_indices[:new_schedules_needed]
                batch.extend(selected)
                # Remove from unshown
                for idx in selected:
                    self.unshown_indices.remove(idx)
                print(f"  Added {new_schedules_needed} new unshown schedules: {selected}")
            else:
                # Not enough unshown schedules
                # Add all remaining unshown
                if self.unshown_indices:
                    batch.extend(self.unshown_indices)
                    print(f"  Added remaining {len(self.unshown_indices)} unshown: {self.unshown_indices}")
                    remaining_needed = new_schedules_needed - len(self.unshown_indices)
                    self.unshown_indices = []
                else:
                    remaining_needed = new_schedules_needed
                
                # Fill remainder with least-shown schedules (excluding current winner)
                if remaining_needed > 0:
                    # Get schedules sorted by appearance count (ascending)
                    candidates = [(i, self.schedule_appearances[i]) 
                                for i in range(self.n_schedules) 
                                if i not in batch]
                    candidates.sort(key=lambda x: (x[1], np.random.random()))  # Sort by appearances, random tiebreak
                    
                    selected = [idx for idx, _ in candidates[:remaining_needed]]
                    batch.extend(selected)
                    print(f"  Added {len(selected)} least-shown schedules: {selected}")
        
        # Ensure we have exactly batch_size schedules
        batch = batch[:self.batch_size]
        
        # Shuffle the batch so winner isn't always first
        np.random.shuffle(batch)
        
        return batch
    
    def update_statistics(self, batch_indices: List[int], winner_idx: int):
        """
        Update win counts and Bradley-Terry scores based on batch result.
        
        Args:
            batch_indices: All indices in the batch
            winner_idx: Index of the winner
        """
        # Update appearance counts
        for idx in batch_indices:
            self.schedule_appearances[idx] += 1
        
        # Update win count for winner
        self.schedule_wins[winner_idx] += 1
        
        # Update current winner for carryover
        self.current_winner_idx = winner_idx
        self.winner_history.append(winner_idx)
        
        # Update Bradley-Terry scores (simplified)
        learning_rate = 0.1
        for idx in batch_indices:
            if idx != winner_idx:
                # Winner score increases, loser score decreases
                p_win = 1 / (1 + np.exp(self.schedule_scores[idx] - self.schedule_scores[winner_idx]))
                self.schedule_scores[winner_idx] += learning_rate * (1 - p_win)
                self.schedule_scores[idx] -= learning_rate * (1 - p_win)
    
    def run_experiment(
        self,
        n_batches: int,
        history_file: str,
        save_interval: int = 5
    ) -> Dict:
        """
        Run batch preference learning experiment with guaranteed winner carryover.
        
        Args:
            n_batches: Number of batches to run
            history_file: File to save history
            save_interval: Save history every N batches
            
        Returns:
            Results dictionary
        """
        print(f"Starting batch experiment with batch_size={self.batch_size}")
        print(f"Total batches to run: {n_batches}")
        print(f"Total schedules available: {self.n_schedules}")
        print(f"Carryover strategy: Previous winner + {self.batch_size - 1} new schedules")
        
        # Initialize history
        history = {
            'metadata': {
                'batch_size': self.batch_size,
                'n_batches': n_batches,
                'n_schedules': self.n_schedules,
                'metric_columns': self.metric_columns,
                'start_time': datetime.now().isoformat(),
                'carryover_strategy': 'winner_plus_new'
            },
            'batch_comparisons': [],
            'batch_summaries': [],
            'schedule_statistics': [],
            'carryover_verification': []  # Track carryover success
        }
        
        # Run batches
        for batch_idx in range(n_batches):
            print(f"\n{'='*60}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print(f"{'='*60}")
            
            # Select next batch
            batch_indices = self.select_next_batch(batch_idx)
            print(f"Selected schedules: {batch_indices}")
            
            # Verify carryover (for batches after the first)
            if batch_idx > 0 and self.winner_history:
                previous_winner = self.winner_history[-1]
                carryover_success = previous_winner in batch_indices
                history['carryover_verification'].append({
                    'batch_num': batch_idx + 1,
                    'previous_winner': previous_winner,
                    'current_batch': batch_indices,
                    'carryover_success': carryover_success
                })
                
                if carryover_success:
                    print(f"✓ Previous winner {previous_winner} successfully carried over")
                else:
                    print(f"✗ ERROR: Previous winner {previous_winner} NOT in current batch!")
            
            # Collect preference
            preference = self.collect_batch_preference(batch_indices)
            
            # Update statistics
            self.update_statistics(batch_indices, preference['winner_idx'])
            
            # Record comparison with full schedule details
            comparison = {
                'batch_num': batch_idx + 1,
                'batch_indices': batch_indices,
                'winner_idx': preference['winner_idx'],
                'choice_letter': preference['choice_letter'],
                'reasoning': preference['reasoning'],
                'timestamp': datetime.now().isoformat(),
                'previous_winner_carried': self.winner_history[-2] if len(self.winner_history) > 1 else None
            }
            
            # Add full schedule details for each option
            batch_schedules = {}
            for i, idx in enumerate(batch_indices):
                letter = chr(ord('A') + i)
                schedule_data = self.schedules_df.iloc[idx].to_dict()
                batch_schedules[f'option_{letter}'] = {
                    'index': idx,
                    'metrics': schedule_data,
                    'wins_at_time': int(self.schedule_wins[idx]),
                    'appearances_at_time': int(self.schedule_appearances[idx]),
                    'win_rate_at_time': float(self.schedule_wins[idx] / max(1, self.schedule_appearances[idx]))
                }
            comparison['batch_schedules'] = batch_schedules
            
            # Add winner's full schedule data
            comparison['winner_schedule'] = self.schedules_df.iloc[preference['winner_idx']].to_dict()
            
            history['batch_comparisons'].append(comparison)
            
            # Record batch summary
            win_rates = np.divide(
                self.schedule_wins,
                np.maximum(self.schedule_appearances, 1)
            )
            top_5 = np.argsort(win_rates)[-5:][::-1]
            
            summary = {
                'batch_num': batch_idx + 1,
                'top_5_indices': top_5.tolist(),
                'top_5_win_rates': win_rates[top_5].tolist(),
                'top_5_appearances': self.schedule_appearances[top_5].tolist(),
                'total_schedules_seen': int(np.sum(self.schedule_appearances > 0)),
                'unshown_remaining': len(self.unshown_indices),
                'current_winner': self.current_winner_idx
            }
            history['batch_summaries'].append(summary)
            
            print(f"\nCurrent top 5 schedules:")
            for i, idx in enumerate(top_5):
                print(f"  {i+1}. Schedule {idx}: "
                      f"Win rate = {win_rates[idx]:.2f} "
                      f"({int(self.schedule_wins[idx])}/{int(self.schedule_appearances[idx])} wins)")
            
            print(f"Unshown schedules remaining: {len(self.unshown_indices)}")
            
            # Save periodically
            if (batch_idx + 1) % save_interval == 0:
                self.save_history(history, history_file)
        
        # Final statistics
        history['metadata']['end_time'] = datetime.now().isoformat()
        
        # Calculate carryover success rate
        if history['carryover_verification']:
            carryover_successes = sum(1 for v in history['carryover_verification'] if v['carryover_success'])
            carryover_rate = carryover_successes / len(history['carryover_verification'])
            history['metadata']['carryover_success_rate'] = carryover_rate
            print(f"\nFinal carryover success rate: {carryover_rate:.1%}")
        
        # Add final schedule statistics
        for idx in range(self.n_schedules):
            if self.schedule_appearances[idx] > 0:
                history['schedule_statistics'].append({
                    'schedule_idx': idx,
                    'wins': int(self.schedule_wins[idx]),
                    'appearances': int(self.schedule_appearances[idx]),
                    'win_rate': self.schedule_wins[idx] / self.schedule_appearances[idx],
                    'bradley_terry_score': float(self.schedule_scores[idx])
                })
        
        # Save final history
        self.save_history(history, history_file)
        
        # Prepare results
        win_rates = np.divide(
            self.schedule_wins,
            np.maximum(self.schedule_appearances, 1)
        )
        final_ranking = np.argsort(win_rates)[::-1]
        
        results = {
            'final_ranking': final_ranking[:10].tolist(),
            'final_win_rates': win_rates[final_ranking[:10]].tolist(),
            'final_bradley_terry_scores': self.schedule_scores[final_ranking[:10]].tolist(),
            'total_batches': n_batches,
            'total_schedules_evaluated': int(np.sum(self.schedule_appearances > 0)),
            'history_file': history_file,
            'carryover_success_rate': history['metadata'].get('carryover_success_rate', 0)
        }
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Top 10 schedules:")
        for i, idx in enumerate(final_ranking[:10]):
            if self.schedule_appearances[idx] > 0:
                print(f"  {i+1}. Schedule {idx}: "
                      f"Win rate = {win_rates[idx]:.2f} "
                      f"({int(self.schedule_wins[idx])}/{int(self.schedule_appearances[idx])} wins), "
                      f"BT score = {self.schedule_scores[idx]:.2f}")
        
        return results
    
    def save_history(self, history: Dict, filename: str):
        """Save history to CSV and JSON files with full schedule details."""
        import json
        
        # Save JSON with all details
        json_file = filename.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        print(f"History saved to {json_file}")