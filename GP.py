from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from Utilitymodel import UtilityModel
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import Matern

import numpy as np
from typing import Optional, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from Utilitymodel import UtilityModel


class GPUtilityModel(UtilityModel):
    """
    GP over scalar utility u(x) trained on per-item pseudo-utilities
    derived from win-rates:
        u_i = logit( clip(wins_i / (wins_i + losses_i)) )

    Speed-ups:
      • Optimize GP hyperparams only on the FIRST fit; then freeze (no optimizer).
      • Train only on items with at least `min_votes`; optionally cap to `max_items`
        most-voted items to keep O(N^3) small.
      • Cache posterior mean/covariance over the full item set for fast sampling.
    """

    def __init__(
        self,
        kernel=None,
        alpha: float = 1e-2,
        normalize_y: bool = True,
        random_state: int = 42,
        eps_clip: float = 1e-3,
        min_votes: int = 3,  # require >= this many total votes to include an item
        max_items: Optional[
            int
        ] = 800,  # cap training set size (most-voted items); None disables
        n_restarts_first_fit: int = 1,  # restarts for first hyperparam fit
        jitter: float = 1e-9,  # added to diag when sampling if needed
    ):
        self.gp = GaussianProcessRegressor(
            kernel=kernel or Matern(nu=2.5),
            alpha=alpha,
            normalize_y=normalize_y,
            random_state=random_state,
        )

        # Data stores
        self._wins: Optional[np.ndarray] = None
        self._losses: Optional[np.ndarray] = None
        self._X_items: Optional[np.ndarray] = None

        # Flags / caches
        self._trained: bool = False
        self._first_fit_done: bool = False
        self._eps = eps_clip
        self._min_votes = int(min_votes)
        self._max_items = max_items if (max_items is None or max_items > 0) else None
        self._n_restarts_first_fit = int(max(0, n_restarts_first_fit))
        self._jitter = float(jitter)

        self._mu_cache: Optional[np.ndarray] = None
        self._cov_cache: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(random_state)

    # ---------------------------
    # Lifecycle / data updates
    # ---------------------------
    def ensure_item_store(self, n_items: int, X_items: np.ndarray) -> None:
        if self._wins is None or len(self._wins) != n_items:
            self._wins = np.zeros(n_items, dtype=float)
            self._losses = np.zeros(n_items, dtype=float)
        self._X_items = np.asarray(X_items, dtype=float)

    def observe_duels(self, duels: List[Tuple[int, int, int]]) -> None:
        """
        duels: list of (i, j, y01) with y01=1 if i beats j, else 0.
        """
        if self._wins is None:
            raise RuntimeError("Call ensure_item_store() before observe_duels().")
        for i, j, y01 in duels:
            if y01 == 1:
                self._wins[i] += 1.0
                self._losses[j] += 1.0
            else:
                self._wins[j] += 1.0
                self._losses[i] += 1.0

    def _pseudo_utilities(self) -> Optional[np.ndarray]:
        """Compute logit(win_rate) per item; returns shape (N,) or None if no labels."""
        if self._wins is None or self._X_items is None:
            return None
        total = self._wins + self._losses
        if not np.any(total > 0):
            return None

        wr = np.zeros_like(total, dtype=float)
        mask_obs = total > 0
        wr[mask_obs] = self._wins[mask_obs] / total[mask_obs]
        wr = np.clip(wr, self._eps, 1.0 - self._eps)
        return np.log(wr / (1.0 - wr))  # logit

    # ---------------------------
    # Training
    # ---------------------------
    def fit_on_duels(self, X_delta: np.ndarray, y01: np.ndarray) -> None:
        """
        Refits the GP using current pseudo-utilities.
        NOTE: X_delta, y01 are unused directly (win/losses are updated via observe_duels()).
        """
        if self._X_items is None:
            return

        u = self._pseudo_utilities()
        if u is None:
            return

        # Select training subset: items with enough votes
        total = self._wins + self._losses
        mask = total >= self._min_votes

        # Optionally cap to the most-voted items to keep matrix small
        if self._max_items is not None and mask.sum() > self._max_items:
            idx_all = np.flatnonzero(mask)
            # pick top by total votes
            top_idx = idx_all[np.argsort(-total[idx_all])[: self._max_items]]
            mask = np.zeros_like(mask, dtype=bool)
            mask[top_idx] = True

        if not np.any(mask):
            return

        X_obs = self._X_items[mask]
        u_obs = u[mask]

        # First fit: optimize hyperparams; later fits: freeze (no optimizer)
        if not self._first_fit_done:
            self.gp.set_params(
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=self._n_restarts_first_fit,
            )
            self.gp.fit(X_obs, u_obs)
            self._first_fit_done = True
        else:
            # Freeze learned hyperparams
            self.gp.set_params(optimizer=None, n_restarts_optimizer=0)
            # Keep the learned kernel_ as-is and refit posterior
            self.gp.kernel_ = self.gp.kernel_
            self.gp.fit(X_obs, u_obs)

        # Cache posterior over the full item set for fast joint sampling
        mu, cov = self.gp.predict(self._X_items, return_cov=True)
        self._mu_cache, self._cov_cache = mu, cov
        self._trained = True

    # ---------------------------
    # Queries
    # ---------------------------
    def posterior_mean_util(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            return np.zeros(X.shape[0], dtype=float)
        # Predict at arbitrary X (no cache assumption here)
        mu, _ = self.gp.predict(np.asarray(X, dtype=float), return_std=True)
        return mu

    def sample_utilities(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        if not self._trained:
            return np.zeros((X.shape[0], n_samples), dtype=float)

        X = np.asarray(X, dtype=float)
        # Fast path: if X matches stored items, reuse cached joint covariance
        if (
            self._X_items is not None
            and X.shape == self._X_items.shape
            and np.allclose(X, self._X_items)
            and (self._mu_cache is not None)
            and (self._cov_cache is not None)
        ):
            mu = self._mu_cache
            cov = self._cov_cache
        else:
            mu, cov = self.gp.predict(X, return_cov=True)

        # Add tiny jitter if needed to ensure PSD for sampling
        try:
            samples = self._rng.multivariate_normal(mu, cov, size=int(n_samples)).T
        except np.linalg.LinAlgError:
            cov_j = cov + self._jitter * np.eye(cov.shape[0])
            samples = self._rng.multivariate_normal(mu, cov_j, size=int(n_samples)).T

        return samples  # (n_items, n_samples)

    def predict_proba_delta(self, delta: np.ndarray) -> float:
        """
        We don't model pairwise prob natively here; acquisition should estimate it
        via sampling the two items' utilities. Returning a neutral 0.5 avoids bias.
        """
        return 0.5

    def ready(self) -> bool:
        return self._trained
