import math
import numpy as np
import pandas as pd
from collections import deque
from .base import VolatilityModelBase
import statsmodels.api as sm

class HARVolModel(VolatilityModelBase):
    """
    HAR (Heterogeneous AutoRegressive) volatility model with periodic refitting.
    """

    def __init__(self,
                 short_window=5,          # seconds
                 med_window=60,           # seconds
                 long_window=600,         # seconds
                 window=600,              # rolling dataset size
                 refit_freq=60,           # refit every X seconds
                 min_window_ratio=0.8):

        # Horizon settings
        self.short_window = short_window
        self.med_window = med_window
        self.long_window = long_window
        self.window = window
        self.refit_freq = refit_freq
        self.min_window_ratio = min_window_ratio

        # Buffers
        self.prev_mid = None
        self.returns = deque(maxlen=self.long_window * 2)
        self.feature_fifo = deque(maxlen=10_000)
        self.rv_history = deque(maxlen=self.window)

        # State
        self._sigma = 0.0
        self._fit_counter = 0
        self._last_fit = None
        self.beta = np.array([-6.5, 0.5, 0.3, 0.2])  # initial guess [β0, β1, β2, β3]

    def update(self, ts: pd.Timestamp, mid: float, bid: float, ask: float):
        if self.prev_mid is None:
            self.prev_mid = mid
            return

        r = math.log((mid + 1e-12) / (self.prev_mid + 1e-12))
        self.prev_mid = mid
        self.returns.append(r)
        if len(self.returns) < self.long_window:
            return

        rv_short = sum(x * x for x in list(self.returns)[-self.short_window:])
        rv_med = sum(x * x for x in list(self.returns)[-self.med_window:])
        rv_long = sum(x * x for x in list(self.returns)[-self.long_window:])

        # Current feature vector (for time t)
        X_now = [math.log(rv_short + 1e-12),
                 math.log(rv_med + 1e-12),
                 math.log(rv_long + 1e-12)]
        self.feature_fifo.append(X_now)

        # Once enough future data has passed, get target (rv_short_future)
        if len(self.feature_fifo) > self.short_window:
            X_past = self.feature_fifo[-(self.short_window + 1)]
            y_future = math.log(rv_short + 1e-12)  # this is future RV for that past feature row
            self.rv_history.append((y_future, *X_past))

        # Wait until we have enough data before refitting
        if len(self.rv_history) < self.window * self.min_window_ratio:
            return

        # Periodic refit
        self._fit_counter += 1
        if self._fit_counter % self.refit_freq != 0:
            return

        df = pd.DataFrame(self.rv_history, columns=["y", "rv_short", "rv_med", "rv_long"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < self.window * self.min_window_ratio:
            return

        # Prepare X, y
        X = sm.add_constant(df[["rv_short", "rv_med", "rv_long"]], has_constant="add")
        y = df["y"]

        try:
            # OLS with mild ridge regularization for numerical stability
            model = sm.OLS(y, X).fit()
            params = model.params.values

            # Ensure 4 coefficients
            if len(params) == 3:
                params = np.insert(params, 0, 0.0)

            # Skip degenerate fits (identity or R² ~ 1)
            if model.rsquared >= 0.999:
                print(f"[HAR REFIT SKIPPED] ts={ts} (R²≈1.000, degenerate)")
                return

            self.beta = params
            self._last_fit = model

            # print(f"[HAR REFIT] ts={ts}, "
            #       f"β₀={self.beta[0]:.4f}, β₁={self.beta[1]:.4f}, "
            #       f"β₂={self.beta[2]:.4f}, β₃={self.beta[3]:.4f}, "
            #       f"R²={model.rsquared:.3f}, n={len(df)}")

        except Exception as e:
            print(f"[HAR REFIT FAILED] ts={ts}: {e}")
            if self._last_fit is not None:
                self.beta = self._last_fit.params.values

        # Predict next volatility
        x_now = np.array([1.0] + X_now)
        log_sigma2 = np.dot(self.beta, x_now)
        self._sigma = math.sqrt(math.exp(log_sigma2))

    def predict(self) -> float:
        return self._sigma
