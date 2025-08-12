from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class UtilityModel(ABC):
    """Abstract interface the acquisition code uses."""

    @abstractmethod
    def fit_on_duels(self, X_delta: np.ndarray, y01: np.ndarray) -> None:
        """Fit/update the model using pairwise deltas and binary outcomes (1 = i beats j)."""

    @abstractmethod
    def posterior_mean_util(self, X: np.ndarray) -> np.ndarray:
        """Return E[u(x)] for each item row in X."""

    @abstractmethod
    def sample_utilities(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Draw joint samples of utilities. Shape: (n_items, n_samples)."""

    @abstractmethod
    def predict_proba_delta(self, delta: np.ndarray) -> float:
        """Return P(i>j | delta)."""

    @abstractmethod
    def ready(self) -> bool:
        """Return True if the model is trained enough to be useful."""
