import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from Utilitymodel import UtilityModel


from typing import Optional
import numpy as np
from numpy.linalg import inv
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


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
            # You could set a flag and skip fitting until you have both classes.
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
