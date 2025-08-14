import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from Utilitymodel import UtilityModel

class GPUtilityModel(UtilityModel):
    def __init__(
        self,
        data_csv_path: str = "data.csv",
        random_state: int = 42,
        eps_clip: float = 1e-3,
        min_votes: int = 1,              # lower this so more than 1 item trains
        max_items: Optional[int] = None, # don't cap early; removes accidental tiny sets
        n_restarts_first_fit: int = 8,
        more_restarts: int = 4,
        jitter: float = 1e-9,
        freeze_n: int = 100,
    ):
        # TIGHT bounds early to prevent "flat everywhere"
        base_kernel = (ConstantKernel(1.0, (1e-2, 1e2)) *
                       Matern(length_scale=0.2, length_scale_bounds=(1e-2, 1.0), nu=2.5) +
                       WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))
        self.gp = GaussianProcessRegressor(
            kernel=base_kernel, alpha=0.0, normalize_y=True, random_state=random_state
        )

        self._wins: Optional[np.ndarray] = None
        self._losses: Optional[np.ndarray] = None
        self._X_items_raw: Optional[np.ndarray] = None
        self._X_items: Optional[np.ndarray] = None

        self._data_csv_path = data_csv_path
        self._col_max: Optional[np.ndarray] = None

        self._trained = False
        self._first_fit_done = False
        self._eps = float(eps_clip)
        self._min_votes = int(min_votes)
        self._max_items = max_items if (max_items is None or max_items > 0) else None
        self._n_restarts_first_fit = int(max(1, n_restarts_first_fit))
        self._more_restarts = int(max(0, more_restarts))
        self._jitter = float(jitter)
        self._freeze_n = int(max(2, freeze_n))

        self._mu_cache: Optional[np.ndarray] = None
        self._cov_cache: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(random_state)

    # ---------- scaling ----------
    def _load_col_max_from_csv(self) -> Optional[np.ndarray]:
        if not os.path.exists(self._data_csv_path): return None
        df = pd.read_csv(self._data_csv_path)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0: return None
        col_max = numeric.max(axis=0).to_numpy(dtype=float)
        col_max[col_max == 0.0] = 1.0
        return col_max

    def _ensure_scaler(self, X: np.ndarray) -> None:
        if self._col_max is None:
            cm = self._load_col_max_from_csv()
            if cm is not None and cm.shape[0] == X.shape[1]:
                self._col_max = cm
            else:
                self._col_max = np.max(np.abs(X), axis=0)
                self._col_max[self._col_max == 0.0] = 1.0

    def _scale(self, X: np.ndarray) -> np.ndarray:
        self._ensure_scaler(X)
        return np.asarray(X, dtype=float) / self._col_max

    # ---------- lifecycle ----------
    def ensure_item_store(self, n_items: int, X_items: np.ndarray) -> None:
        if self._wins is None or len(self._wins) != n_items:
            self._wins = np.zeros(n_items, dtype=float)
            self._losses = np.zeros(n_items, dtype=float)
        self._X_items_raw = np.asarray(X_items, dtype=float)
        self._X_items = self._scale(self._X_items_raw)

        # verify X is not collapsed
        self._debug_X()

        # set ARD kernel with dim-sized length_scales
        d = self._X_items.shape[1]
        ls0 = np.clip(self._X_items.std(axis=0), 1e-2, 1.0)  # init by per-dim std in [1e-2,1]
        k = (ConstantKernel(1.0, (1e-2, 1e2)) *
             Matern(length_scale=ls0, length_scale_bounds=(1e-2, 1.0), nu=2.5) +
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))
        self.gp.set_params(kernel=k)

    def observe_duels(self, duels: List[Tuple[int, int, int]]) -> None:
        if self._wins is None: raise RuntimeError("Call ensure_item_store() first.")
        for i, j, y01 in duels:
            if y01 == 1:
                self._wins[i] += 1.0; self._losses[j] += 1.0
            else:
                self._wins[j] += 1.0; self._losses[i] += 1.0

    def _pseudo_utilities(self) -> Optional[np.ndarray]:
        if self._wins is None or self._X_items is None: return None
        total = self._wins + self._losses
        if not np.any(total > 0): return None
        wr = np.zeros_like(total, dtype=float)
        mask_obs = total > 0
        # Laplace smoothing
        wr[mask_obs] = (self._wins[mask_obs] + 1.0) / (total[mask_obs] + 2.0)
        wr = np.clip(wr, self._eps, 1.0 - self._eps)
        return np.log(wr / (1.0 - wr))

    # ---------- training ----------
    def fit_on_duels(self, X_delta: np.ndarray, y01: np.ndarray) -> None:
        if self._X_items is None: return
        u = self._pseudo_utilities()
        if u is None: return

        total = self._wins + self._losses
        mask = total >= self._min_votes

        # ensure we have >1 DISTINCT training point and non-constant targets
        if np.count_nonzero(mask) < 2:
            print("[GP] Not enough train items (mask sum < 2). Skipping fit.")
            return
        X_obs = self._X_items[mask]
        u_obs = u[mask]

        # drop duplicate rows to avoid singular covariance
        _, uniq_idx = np.unique(X_obs, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        X_obs = X_obs[uniq_idx]
        u_obs = u_obs[uniq_idx]

        if X_obs.shape[0] < 2:
            print("[GP] Only one unique X row after dedup. Skipping fit.")
            return
        if np.allclose(u_obs, u_obs[0]):
            print("[GP] Targets constant after masking (u std≈0). Skipping fit.")
            return

        n_train = X_obs.shape[0]
        # keep optimizing until freeze_n, then freeze
        if not self._first_fit_done or n_train < self._freeze_n:
            self.gp.set_params(optimizer="fmin_l_bfgs_b",
                               n_restarts_optimizer=self._n_restarts_first_fit if not self._first_fit_done else self._more_restarts)
        else:
            self.gp.set_params(optimizer=None, n_restarts_optimizer=0)

        self.gp.fit(X_obs, u_obs)
        self._first_fit_done = True

        mu, cov = self.gp.predict(self._X_items, return_cov=True)
        self._mu_cache, self._cov_cache = mu, cov
        self._trained = True

        self._debug_after_fit(X_obs, u_obs, n_train)

    # ---------- queries ----------
    def posterior_mean_util(self, X: np.ndarray) -> np.ndarray:
        if not self._trained: return np.zeros(X.shape[0], dtype=float)
        Xs = self._scale(np.asarray(X, dtype=float))
        mu, _ = self.gp.predict(Xs, return_std=True)
        return mu

    def sample_utilities(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        if not self._trained: return np.zeros((X.shape[0], n_samples), dtype=float)
        Xs = self._scale(np.asarray(X, dtype=float))
        if (self._X_items is not None and Xs.shape == self._X_items.shape
            and np.allclose(Xs, self._X_items)
            and (self._mu_cache is not None) and (self._cov_cache is not None)):
            mu, cov = self._mu_cache, self._cov_cache
        else:
            mu, cov = self.gp.predict(Xs, return_cov=True)
        try:
            S = self._rng.multivariate_normal(mu, cov, size=int(n_samples)).T
        except np.linalg.LinAlgError:
            cov_j = cov + self._jitter * np.eye(cov.shape[0])
            S = self._rng.multivariate_normal(mu, cov_j, size=int(n_samples)).T
        return S

    def predict_proba_delta(self, delta: np.ndarray) -> float:
        return 0.5
    def predict_mu_std(self, X: np.ndarray):
        """Scaled prediction wrapper so callers never bypass the scaler."""
        if not self._trained:
            n = X.shape[0]
            return np.zeros(n, dtype=float), np.ones(n, dtype=float)
        Xs = self._scale(np.asarray(X, dtype=float))
        mu, std = self.gp.predict(Xs, return_std=True)
        return mu, std
    def ready(self) -> bool:
        return self._trained

    # ---------- debug ----------
    def _debug_X(self):
        X = self._X_items
        if X is None: return
        n, d = X.shape
        uniq = np.unique(X, axis=0).shape[0]
        col_std = X.std(axis=0)
        print(f"[X] N={n}, d={d}, unique_rows={uniq}")
        with np.printoptions(precision=4, suppress=True, threshold=10):
            print("[X] per-dim std (first 8):", col_std[:8], "...")
        if np.any(col_std < 1e-8):
            bad = np.where(col_std < 1e-8)[0]
            print(f"[X] WARNING: near-constant columns after scaling at dims {bad[:10]}")

    def _debug_after_fit(self, X_obs, u_obs, n_train):
        print(f"[GP fit] n_train={n_train}")
        print("[GP fit] kernel_:", self.gp.kernel_)
        print("[GP fit] u_obs: mean=%.4f std=%.4f min=%.4f max=%.4f" %
              (u_obs.mean(), u_obs.std(), u_obs.min(), u_obs.max()))
        # quick look at predictive stats on the bank
        mu, std = self.gp.predict(self._X_items, return_std=True)
        print("[GP pred] mu:  mean=%.4f std=%.4f | mu[:5]=" % (mu.mean(), mu.std()), mu[:5])
        print("[GP pred] s:   mean=%.4f std=%.4f | s[:5]=" % (std.mean(), std.std()), std[:5])
