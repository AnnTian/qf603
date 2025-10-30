import numpy as np
from arch import arch_model
from volmodels import VolatilityModelBase

class GARCHVolModel(VolatilityModelBase):
    """
    Real-time GARCH volatility model with periodic refitting and cached predictions.
    Inspired by Code 2, but based on your Code 1 structure.
    """

    def __init__(self, window=1000, p=3, q=1, refit_freq=10, dist='t',
                 fast_mode=True, min_window_ratio=0.8):
        """
        Args:
            window: number of past returns to use for GARCH fitting
            p, q: GARCH(p, q) order
            refit_freq: frequency of model refitting (e.g. every 10 updates)
            dist: residual distribution ('normal', 't', 'skewt')
            fast_mode: if True, uses fewer iterations and looser tolerance
            min_window_ratio: minimum ratio of valid samples to attempt a refit
        """
        self.window = window
        self.p = p
        self.q = q
        self.refit_freq = refit_freq
        self.dist = dist
        self.fast_mode = fast_mode
        self.min_window_ratio = min_window_ratio

        self.returns = []
        self.prev_mid = None
        self._fit_counter = 0
        self._last_fit = None
        self._sigma = 0.0  # cached volatility estimate

    # -----------------------------
    # Update logic
    # -----------------------------
    def update(self, ts, mid, bid, ask):
        """Update model with a new tick and periodically refit GARCH."""
        if self.prev_mid is None or mid <= 0:
            self.prev_mid = mid
            return

        # Compute log return
        r = np.log(mid / self.prev_mid)
        self.prev_mid = mid
        self.returns.append(r)

        # Keep rolling window
        if len(self.returns) > self.window:
            self.returns.pop(0)

        # Skip until enough data collected
        if len(self.returns) < self.window * self.min_window_ratio:
            return

        # Increment update counter
        self._fit_counter += 1
        if self._fit_counter % self.refit_freq != 0:
            return  # only refit periodically

        # Prepare data
        returns_array = np.array(self.returns[-self.window:])
        returns_array = returns_array[np.isfinite(returns_array)]
        if len(returns_array) < self.window * self.min_window_ratio:
            return

        # Demean data for stability
        returns_array -= returns_array.mean()

        # Check variance
        var = np.var(returns_array, ddof=1)
        if var <= 1e-12:
            self._sigma = np.sqrt(var)
            return

        # -----------------------------
        # Fit GARCH model
        # -----------------------------
        model = arch_model(
            returns_array,
            mean='Zero',
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=True
        )

        options = {'maxiter': 150 if self.fast_mode else 500,
                   'ftol': 1e-4 if self.fast_mode else 1e-6}

        start_params = None
        if self._last_fit is not None:
            try:
                start_params = self._last_fit.params.values
            except Exception:
                start_params = None

        try:
            fitted = model.fit(
                disp='off',
                show_warning=False,
                update_freq=0,
                options=options,
                starting_values=start_params
            )

            if not fitted.converged:
                raise RuntimeError("GARCH model failed to converge.")

            self._last_fit = fitted
            forecast = fitted.forecast(horizon=1, reindex=False)
            sigma = np.sqrt(forecast.variance.values[-1, 0])
            if np.isfinite(sigma):
                self._sigma = float(sigma)
            else:
                raise ValueError("Invalid sigma (NaN/inf)")

        except Exception as e:
            # On fitting failure: fallback to previous fit or std
            if self._last_fit is not None:
                forecast = self._last_fit.forecast(horizon=1, reindex=False)
                self._sigma = float(np.sqrt(forecast.variance.values[-1, 0]))
            else:
                self._sigma = float(np.std(returns_array))

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self):
        """Return the most recent volatility estimate (cached)."""
        return self._sigma
