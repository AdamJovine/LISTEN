import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import json
import csv
from datetime import datetime
from LISTEN import CustomDuelingBanditOptimizer
from oracle import Oracle
from prompt import PromptTemplate
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
        prompt_template: PromptTemplate = PromptTemplate(), 
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
        prompt = self.prompt_template.format(
            schedule_a=sched_a,
            schedule_b=sched_b
        )

        # Call LLM (implement based on your LLM client)
        desicion,reasoning = self.llm_client.call_oracle(prompt, sched_a , sched_b )
        print('reasoning, ' , reasoning )
        print('decision ' , desicion)
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
                winner = self.collect_llm_preference(
                    idx_a, idx_b, 
                    with_history=with_reasoning_history
                )
                
                comparison = {
                    'batch': batch_idx,
                    'option_a': idx_a,
                    'option_b': idx_b,
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
                top_5 = np.argsort(utilities)[-5:][::-1]
                
                batch_result = {
                    'batch': batch_idx,
                    'top_5_indices': top_5.tolist(),
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
                'batch', 'option_a', 'option_b', 'winner', 'timestamp'
            ])
            writer.writeheader()
            writer.writerows(history['comparisons'])
        
        print(f"History saved to {filename} and {json_file}")

