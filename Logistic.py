import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from Utilitymodel import UtilityModel


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class LinearLogisticModel(UtilityModel):
    """Linear utility u(x)=theta^T x with logistic pairwise likelihood."""

    def __init__(self, C: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.C = C
        self.theta_mean: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self._lr = LogisticRegression(C=C)
        self._trained = False
        self._rng = rng or np.random.default_rng()

    def fit_on_duels(self, X_delta: np.ndarray, y01: np.ndarray) -> None:
        if X_delta.size == 0:
            return
        self._lr.fit(X_delta, y01)
        theta_hat = self._lr.coef_.ravel()
        # Laplace covariance at MAP
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
        if not self._trained or self.theta_mean is None:
            return np.zeros(X.shape[0])
        return X @ self.theta_mean

    def _sample_thetas(self, n_samples: int) -> np.ndarray:
        if not self._trained or self.theta_mean is None:
            return np.zeros((n_samples, 0))
        if self.Sigma is None:
            return np.tile(self.theta_mean, (n_samples, 1))
        return self._rng.multivariate_normal(
            self.theta_mean, self.Sigma, size=n_samples
        )

    def sample_utilities(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        thetas = self._sample_thetas(n_samples)  # (S, d)
        return X @ thetas.T  # (n_items, S)

    def predict_proba_delta(self, delta: np.ndarray) -> float:
        if not self._trained or self.theta_mean is None:
            return 0.5
        return float(expit(delta @ self.theta_mean))

    def ready(self) -> bool:
        return self._trained
