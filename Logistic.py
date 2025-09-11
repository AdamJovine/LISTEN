import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from typing import Optional, List, Tuple


class LinearLogisticModel:
    """Linear utility u(x)=theta^T x with Bradley-Terry pairwise likelihood."""

    def __init__(self, C: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.C = C
        self.theta_mean: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self._lr = LogisticRegression(C=C)
        self._trained = False
        self._rng = rng or np.random.default_rng()

    def fit_on_duels(self, X_delta: np.ndarray, y01: np.ndarray) -> None:
        """Fit Bradley-Terry model on pairwise comparisons."""
        if X_delta.size == 0:
            return

        # Ensure 2D features and 1D labels
        if X_delta.ndim == 1:
            X_delta = X_delta.reshape(-1, 1)
        y01 = y01.ravel()

        # If all labels identical, flip exactly ONE example (and negate its delta)
        if y01.size >= 2 and (np.all(y01 == 0) or np.all(y01 == 1)):
            # pick the row with the largest norm (more informative flip); fallback to 0
            try:
                idx = int(np.argmax(np.linalg.norm(X_delta, axis=1)))
            except Exception:
                idx = 0
            y01 = y01.copy()
            X_delta = X_delta.copy()
            y01[idx] = 1 - y01[idx]
            X_delta[idx] = -X_delta[idx]

        # If there's still only one unique class (e.g., n==1), bail gracefully
        if np.unique(y01).size < 2:
            return

        self._lr.fit(X_delta, y01)
        theta_hat = self._lr.coef_.ravel()

        # Laplace approximation for posterior covariance at MAP
        p = expit(X_delta @ theta_hat)
        W = p * (1 - p)
        lam = 1.0 / max(self.C, 1e-9)
        Xw = X_delta * W[:, None]
        H = X_delta.T @ Xw + lam * np.eye(X_delta.shape[1])
        try:
            Sigma = inv(H)
        except np.linalg.LinAlgError:
            Sigma = inv(H + 1e-6 * np.eye(H.shape[1]))

        self.theta_mean, self.Sigma = theta_hat, Sigma
        self._trained = True

    def posterior_mean_util(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior mean utilities u(x) = theta^T x"""
        if not self._trained or self.theta_mean is None:
            return np.zeros(X.shape[0])
        return X @ self.theta_mean

    def posterior_std_util(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior std of utilities"""
        if not self._trained or self.Sigma is None:
            return np.ones(X.shape[0])  # High uncertainty if not trained
        # Var[u(x)] = x^T Sigma x
        var = np.sum((X @ self.Sigma) * X, axis=1)
        return np.sqrt(np.maximum(var, 0))

    def _sample_thetas(self, n_samples: int) -> np.ndarray:
        """Sample from posterior distribution of theta"""
        if not self._trained or self.theta_mean is None:
            d = self.Sigma.shape[0] if self.Sigma is not None else 1
            return np.zeros((n_samples, d))
        if self.Sigma is None:
            return np.tile(self.theta_mean, (n_samples, 1))
        return self._rng.multivariate_normal(
            self.theta_mean, self.Sigma, size=n_samples
        )

    def sample_utilities(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample utilities from posterior"""
        thetas = self._sample_thetas(n_samples)  # (S, d)
        return X @ thetas.T  # (n_items, S)

    def predict_proba(self, winner_idx: int, challenger_idx: int, feat: np.ndarray) -> float:
        """P(challenger beats winner) under Bradley-Terry model"""
        if not self._trained or self.theta_mean is None:
            return 0.5
        
        # Compute utilities
        u_winner = feat[winner_idx] @ self.theta_mean
        u_challenger = feat[challenger_idx] @ self.theta_mean
        
        # P(challenger beats winner) = σ(u(challenger) - u(winner))
        return float(expit(u_challenger - u_winner))

    def predict_proba_delta(self, delta: np.ndarray) -> float:
        """P(A beats B) given delta = features(A) - features(B)"""
        if not self._trained or self.theta_mean is None:
            return 0.5
        return float(expit(delta @ self.theta_mean))

    def compute_score(self, winner_idx: int, challenger_idx: int, feat: np.ndarray, 
                     acquisition: str = "eubo", n_samples: int = 100) -> float:
        """
        Compute acquisition score for comparing challenger to winner.
        
        Args:
            winner_idx: Index of current winner/champion
            challenger_idx: Index of challenger
            feat: Feature matrix for all items
            acquisition: Type of acquisition function
                - "eubo": Expected Utility of Best Option
                - "ucb": Upper Confidence Bound
                - "thompson": Thompson sampling score
                - "info_gain": Expected information gain
        
        Returns:
            Acquisition score (higher is better)
        """
        if not self._trained:
            return self._rng.random()  # Random if not trained
        
        # Get features
        x_winner = feat[winner_idx]
        x_challenger = feat[challenger_idx]
        
        # Compute posterior mean and std of utilities
        u_winner_mean = x_winner @ self.theta_mean
        u_challenger_mean = x_challenger @ self.theta_mean
        
        if acquisition == "eubo":
            # Expected Utility of Best Option
            # Sample utilities and compute P(challenger is best)
            u_samples = self.sample_utilities(feat[[winner_idx, challenger_idx]], n_samples)
            p_challenger_best = np.mean(u_samples[1] > u_samples[0])
            
            # EUBO = P(challenger wins) * E[u(challenger)] + P(winner wins) * E[u(winner)]
            score = p_challenger_best * u_challenger_mean + (1 - p_challenger_best) * u_winner_mean
            
        elif acquisition == "ucb":
            # Upper Confidence Bound for challenger
            u_challenger_std = self.posterior_std_util(x_challenger.reshape(1, -1))[0]
            beta = 2.0  # Exploration parameter
            score = u_challenger_mean + beta * u_challenger_std
            
        elif acquisition == "thompson":
            # Thompson sampling: sample one utility and use it
            theta_sample = self._sample_thetas(1)[0]
            score = x_challenger @ theta_sample
            
        elif acquisition == "info_gain":
            # Expected information gain from this comparison
            # Approximate using entropy of predicted outcome
            p_win = self.predict_proba(winner_idx, challenger_idx, feat)
            
            # Entropy of outcome
            entropy = -p_win * np.log(p_win + 1e-10) - (1-p_win) * np.log(1-p_win + 1e-10)
            
            # Weight by utility to prefer high-value comparisons
            score = entropy * (1 + u_challenger_mean)
        elif acquisition == "random":
            score = self._rng.random()
        else:
            # Default: use posterior mean utility
            score = u_challenger_mean
        
        return float(score)

    def ready(self) -> bool:
        return self._trained


class DuelingBanditSelector:
    """Selector class using Bradley-Terry model for dueling bandits."""
    
    def __init__(self, all_idx, feat, utility_model, compared_pairs, 
                 previous_winners, acquisition: str = "eubo"):
        self.all_idx = all_idx
        self.feat = feat
        self.utility_model = utility_model
        self.compared_pairs = compared_pairs
        self.previous_winners = previous_winners
        self.acquisition = acquisition

    def _select_batch_with_model(self) -> List[Tuple[int, int]]:
        """
        Select batch of comparisons using Bradley-Terry model with acquisition function.
        """
        print(f"Previous winners: {self.previous_winners}")

        if self.previous_winners:
            candidate_pairs: List[Tuple[int, int]] = []
            
            for winner in self.previous_winners:
                # Get all possible challengers (not compared yet)
                challengers_all = [
                    j for j in self.all_idx
                    if j != winner
                    and ((min(winner, j), max(winner, j)) not in self.compared_pairs)
                ]
                
                if not challengers_all:
                    continue
                
                # Sample subset for efficiency
                challengers = np.random.choice(
                    challengers_all, 
                    size=min(500, len(challengers_all)),
                    replace=False
                ).tolist()

                # Score each potential challenger
                scored = []
                for challenger in challengers:
                    score = self.utility_model.compute_score(
                        winner_idx=winner,
                        challenger_idx=challenger,
                        feat=self.feat,
                        acquisition=self.acquisition
                    )
                    scored.append(((winner, challenger), score))
                
                # Select best challenger for this winner
                scored.sort(key=lambda t: t[1], reverse=True)
                if scored:
                    best_pair = scored[0][0]
                    candidate_pairs.append(best_pair)
                    
                    # Optional: print diagnostics
                    if len(scored) > 1:
                        print(f"Winner {winner}: Best challenger {best_pair[1]} "
                              f"with score {scored[0][1]:.3f} "
                              f"(range: {scored[-1][1]:.3f} to {scored[0][1]:.3f})")
            
            if candidate_pairs:
                return candidate_pairs

        # Fallback: if model not ready or no winners, use random selection
        if not self.utility_model.ready():
            return self._select_initial_batch()
        
        # If we get here, select random pairs
        return self._select_initial_batch()

    def _select_initial_batch(self):
        """Initial random batch selection (implement as needed)"""
        # This is a placeholder - implement your initial selection logic
        pairs = []
        n_pairs = min(len(self.previous_winners), 10) if self.previous_winners else 10
        
        for _ in range(n_pairs):
            i, j = np.random.choice(self.all_idx, size=2, replace=False)
            if (min(i, j), max(i, j)) not in self.compared_pairs:
                pairs.append((i, j))
        
        return pairs