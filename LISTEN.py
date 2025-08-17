import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import random
from Logistic import LinearLogisticModel 
from oss import LocalGroqLikeClient
from oracle import Oracle
@dataclass
class ComparisonResult:
    """Store result of a pairwise comparison"""
    option_a: int
    option_b: int
    winner: int  # Should be either option_a or option_b
    features_a: np.ndarray
    features_b: np.ndarray


class DuelingBanditOptimizer:
    """
    Main optimizer that manages the dueling bandit process.
    Trains a new model after each batch of preferences.
    """
    
    def __init__(
        self,
        all_options: List[int],
        features: np.ndarray,
        client:Oracle  , 
        batch_size: int = 10,
        n_champions: int = 5,
        acquisition: str = "eubo",
        C: float = 1.0,
        random_seed: Optional[int] = None, 

    ):
        """
        Args:
            all_options: List of all option indices
            features: Feature matrix (n_options x n_features)
            batch_size: Number of comparisons per batch
            n_champions: Number of top options to maintain as champions
            acquisition: Acquisition function type
            C: Regularization parameter for logistic regression
            random_seed: Random seed for reproducibility
        """
        self.all_options = all_options
        self.features = features
        self.batch_size = batch_size
        self.n_champions = n_champions
        self.acquisition = acquisition
        self.C = C
        self.llm_client = client
        
        # Set random seeds
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize tracking
        self.compared_pairs: Set[Tuple[int, int]] = set()
        self.comparison_history: List[ComparisonResult] = []
        self.current_champions: List[int] = []
        
        # Initialize model (will be retrained each batch)
        self.model = LinearLogisticModel(C=self.C)
        
        # Track utilities over time
        self.utility_history: List[np.ndarray] = []

    def _update_champions(self):
        """Update the list of champion options based on current model."""
        if not self.model.ready():
            # Random champions if model not ready
            self.current_champions = random.sample(
                self.all_options, 
                min(self.n_champions, len(self.all_options))
            )
        else:
            # Get utilities for all options
            utilities = self.model.posterior_mean_util(self.features)
            
            # Select top-k as champions
            top_indices = np.argsort(utilities)[-self.n_champions:][::-1]
            self.current_champions = [self.all_options[i] for i in top_indices]
    
    def _train_model_on_all_data(self):
        """Train model on all historical comparison data."""
        if not self.comparison_history:
            return
        
        # Prepare training data from all comparisons
        X_deltas = []
        y_outcomes = []
        
        for comp in self.comparison_history:
            # Feature difference: winner - loser
            if comp.winner == comp.option_a:
                delta = comp.features_a - comp.features_b
                y = 1  # A beat B
            else:
                delta = comp.features_b - comp.features_a
                y = 1  # B beat A (but we flipped the features)
            
            X_deltas.append(delta)
            y_outcomes.append(y)
        
        # Stack into arrays
        X_delta = np.vstack(X_deltas)
        y = np.array(y_outcomes)
        
        # Train new model from scratch on all data
        self.model = LinearLogisticModel(C=self.C)
        self.model.fit_on_duels(X_delta, y)
        
        # Store utility estimates
        if self.model.ready():
            utilities = self.model.posterior_mean_util(self.features)
            self.utility_history.append(utilities)
    
    def _select_next_batch(self) -> List[Tuple[int, int]]:
        """Select next batch of pairs to compare."""
        pairs = []
        
        if not self.current_champions:
            # Initial random selection
            available_pairs = [
                (i, j) for i in self.all_options 
                for j in self.all_options 
                if i < j and (i, j) not in self.compared_pairs
            ]
            
            if available_pairs:
                n_select = min(self.batch_size, len(available_pairs))
                pairs = random.sample(available_pairs, n_select)
        
        else:
            # Select challengers for each champion
            
            for champion in self.current_champions[:self.batch_size]:
                print('champion' , champion)
                # Get all possible challengers
                challengers = [
                    j for j in self.all_options
                    if j != champion
                    and (min(champion, j), max(champion, j)) not in self.compared_pairs
                ]
                
                if not challengers:
                    continue
                
                if not self.model.ready():
                    # Random challenger if model not ready
                    challenger = random.choice(challengers)
                    pairs.append((champion, challenger))
                
                else:
                    # Score all challengers using acquisition function
                    scores = []
                    
                    # Sample subset for efficiency
                    sample_size = min(1000, len(challengers))
                    challenger_sample = random.sample(challengers, sample_size)
                    
                    for challenger in challenger_sample:
                        score = self.model.compute_score(
                            winner_idx=champion,
                            challenger_idx=challenger,
                            feat=self.features,
                            acquisition=self.acquisition
                        )
                        #print('cham ' , champion)
                        #print('chal , ' , challenger)
                        #print('score , ' , score,  )
                        scores.append((challenger, score))
                    # Select best challenger
                    if scores:
                        best_challenger = max(scores, key=lambda x: x[1])[0]
                        print('cha and base , ' , champion , best_challenger) 
                        pairs.append((champion, best_challenger))

                if len(pairs) >= self.batch_size:
                    break
        
        return pairs[:self.batch_size]
    
    def optimize(self, n_iterations: int) -> Dict:
        """
        Run the optimization process for n_iterations.
        Each iteration selects a batch, collects preferences, and retrains the model.
        
        Returns:
            Dictionary with optimization results and history
        """
        results = {
            'iterations': [],
            'final_champions': [],
            'final_utilities': None,
            'comparison_count': 0
        }
        
        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            # Select next batch of comparisons
            pairs = self._select_next_batch()
            
            if not pairs:
                print("No more pairs to compare!")
                break
            
            print(f"Selected {len(pairs)} pairs for comparison")
            
            # Collect preferences
            batch_results = self._collect_preferences(pairs)
            self.comparison_history.extend(batch_results)
            
            # Train new model on ALL historical data
            self._train_model_on_all_data()
            
            # Update champions based on new model
            self._update_champions()
            
            # Log iteration results
            iter_info = {
                'iteration': iteration + 1,
                'pairs_compared': len(pairs),
                'total_comparisons': len(self.comparison_history),
                'current_champions': self.current_champions.copy(),
                'model_trained': self.model.ready()
            }
            
            if self.model.ready():
                # Get current utility estimates
                utilities = self.model.posterior_mean_util(self.features)
                uncertainties = self.model.posterior_std_util(self.features)
                
                # Find top options
                top_5_idx = np.argsort(utilities)[-5:][::-1]
                
                iter_info['top_5_options'] = top_5_idx.tolist()
                iter_info['top_5_utilities'] = utilities[top_5_idx].tolist()
                iter_info['top_5_uncertainties'] = uncertainties[top_5_idx].tolist()
                
                print(f"Top 5 options: {top_5_idx}")
                print(f"Their utilities: {utilities[top_5_idx].round(3)}")
                print(f"Their uncertainties: {uncertainties[top_5_idx].round(3)}")
            
            results['iterations'].append(iter_info)
            
            # Check convergence (optional)
            if iteration > 5 and self._check_convergence():
                print("Converged early!")
                break
        
        # Final results
        if self.model.ready():
            final_utilities = self.model.posterior_mean_util(self.features)
            results['final_utilities'] = final_utilities
            results['final_champions'] = self.current_champions
            results['comparison_count'] = len(self.comparison_history)
            
            print(f"\n--- Final Results ---")
            print(f"Total comparisons made: {len(self.comparison_history)}")
            print(f"Final champions: {self.current_champions}")
            
            # Show top 10 options
            top_10_idx = np.argsort(final_utilities)[-10:][::-1]
            print(f"\nTop 10 options by utility:")
            for rank, idx in enumerate(top_10_idx):
                print(f"  {rank+1}. Option {idx}: utility = {final_utilities[idx]:.3f}")
        
        return results
    
    def _check_convergence(self, threshold: float = 0.01) -> bool:
        """Check if the optimization has converged."""
        if len(self.utility_history) < 3:
            return False
        
        # Check if utilities have stabilized
        recent_utils = self.utility_history[-3:]
        diffs = [np.max(np.abs(recent_utils[i] - recent_utils[i-1])) 
                 for i in range(1, len(recent_utils))]
        
        return all(d < threshold for d in diffs)




class CustomDuelingBanditOptimizer(DuelingBanditOptimizer):
    """
    Extended optimizer that uses custom preference collection.
    """
    
    def __init__(self, preference_collector, **kwargs):
        super().__init__(**kwargs)
        self.preference_collector = preference_collector
    
    def select_next_batch(self) -> List[Tuple[int, int]]:
        """Public method to get next batch without collecting preferences."""
        return self._select_next_batch()
    
    def update_with_comparisons(self, comparisons: List[Dict]):
        """Update model with collected comparisons."""
        for comp in comparisons:
            # Create ComparisonResult
            result = ComparisonResult(
                option_a=comp['option_a'],
                option_b=comp['option_b'],
                winner=comp['winner'],
                features_a=self.features[comp['option_a']],
                features_b=self.features[comp['option_b']]
            )
            self.comparison_history.append(result)
            
            # Mark as compared
            pair = (min(comp['option_a'], comp['option_b']), 
                   max(comp['option_a'], comp['option_b']))
            self.compared_pairs.add(pair)
        
        # Retrain model
        self._train_model_on_all_data()
        self._update_champions()
