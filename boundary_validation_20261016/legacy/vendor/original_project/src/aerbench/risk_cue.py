from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except Exception:  # pragma: no cover
    LassoCV = None
    StandardScaler = None
    make_pipeline = None


@dataclass
class RiskCueConfig:
    max_lag: int = 16
    horizon_steps: int = 8
    min_samples: int = 120
    window: int = 360
    refit_every: int = 8
    random_state: int = 1


class OnlineRiskCue:
    """Interpretable forward-looking NH4 risk cue using online lagged Lasso.

    It uses only online-visible quantities: measured DO/NH4, previous implemented
    aeration, and current influent descriptors. The cue is a prediction/early-warning
    signal, not a causal estimator.
    """

    def __init__(self, cfg: RiskCueConfig | None = None):
        self.cfg = cfg or RiskCueConfig()
        self.history: List[Dict[str, float]] = []
        self.model = None
        self.feature_names: List[str] = []
        self.last_fit_step = -10**9
        self.last_prediction = np.nan

    def update_history(self, row: Dict[str, float]) -> None:
        self.history.append(row)
        if len(self.history) > self.cfg.window + self.cfg.horizon_steps + self.cfg.max_lag + 10:
            self.history = self.history[-(self.cfg.window + self.cfg.horizon_steps + self.cfg.max_lag + 10):]

    def _build_feature_vector_at(self, idx: int) -> np.ndarray:
        names = []
        values = []
        for var in ["meas_NH4", "meas_DO", "u", "in_COD", "in_NH4"]:
            for lag in [1, 2, 4, 8, self.cfg.max_lag]:
                j = idx - lag
                if j < 0:
                    val = self.history[0].get(var, 0.0)
                else:
                    val = self.history[j].get(var, 0.0)
                names.append(f"{var}_lag{lag}")
                values.append(val)
        self.feature_names = names
        return np.array(values, dtype=float)

    def _training_data(self):
        n = len(self.history)
        end = n - self.cfg.horizon_steps
        start = max(self.cfg.max_lag, end - self.cfg.window)
        X, y = [], []
        for i in range(start, end):
            X.append(self._build_feature_vector_at(i))
            y.append(self.history[i + self.cfg.horizon_steps]["meas_NH4"])
        if not X:
            return np.empty((0, 1)), np.empty((0,))
        return np.vstack(X), np.array(y, dtype=float)

    def maybe_fit(self, step: int) -> None:
        if LassoCV is None:
            return
        if step - self.last_fit_step < self.cfg.refit_every:
            return
        X, y = self._training_data()
        if len(y) < self.cfg.min_samples or np.nanstd(y) < 1e-9:
            return
        alphas = np.logspace(-3, 0.5, 20)
        self.model = make_pipeline(
            StandardScaler(),
            LassoCV(alphas=alphas, cv=3, random_state=self.cfg.random_state, max_iter=20000),
        )
        self.model.fit(X, y)
        self.last_fit_step = step

    def predict(self, step: int) -> float:
        if len(self.history) <= self.cfg.max_lag + 1:
            self.last_prediction = np.nan
            return np.nan
        self.maybe_fit(step)
        current_nh4 = self.history[-1].get("meas_NH4", np.nan)
        if self.model is None:
            self.last_prediction = current_nh4
            return float(current_nh4)
        x = self._build_feature_vector_at(len(self.history) - 1).reshape(1, -1)
        pred = float(self.model.predict(x)[0])
        self.last_prediction = pred
        return pred

    def coefficients(self) -> Dict[str, float]:
        if self.model is None or not hasattr(self.model[-1], "coef_"):
            return {}
        return dict(zip(self.feature_names, map(float, self.model[-1].coef_)))
