from arch import arch_model
import math
import numpy as np 
import pandas as pd 
from .base import VolatilityModelBase

class GARCHVolModel(VolatilityModelBase):
    """
    GARCH volatility model for real-time volatility estimation.
    Uses a rolling window of returns to fit GARCH(3,1) model.
    """
    
    def __init__(self, window: int = 600, p: int = 3, q: int = 1, refit_freq: int = 10, dist: str = 't',
                 fast_mode: bool = True, min_window_ratio: float = 0.8):
        """
        Args:
            window: number of past returns to use for GARCH fitting
            p: number of ARCH terms
            q: number of GARCH terms
            refit_freq: frequency of model refitting 
            dist: t distribution for residuals (heavy tails)
            fast_mode: use faster but less accurate fitting options
            min_window_ratio: minimum ratio of valid data required for refitting
        """
        self.window = window
        self.p = p
        self.q = q
        self.refit_freq = refit_freq
        self.dist = dist
        self.fast_mode = fast_mode
        self.min_window_ratio = min_window_ratio
        self.prev_mid = None
        self.returns = []
        self._sigma = 0.0
        self._fit_counter = 0
        self._last_fit = None

        
    def update(self, ts: pd.Timestamp, mid: float, bid: float, ask: float):
        if self.prev_mid is None:
            self.prev_mid = mid
            return
            
        # log return (pct)
        r = np.log(mid/self.prev_mid)
        self.returns.append(r)
        self.prev_mid = mid
            
        if len(self.returns) < self.window:
            return
        
        self._fit_counter += 1
        if self._fit_counter % self.refit_freq != 0:
            return
        
        returns_array = np.array(self.returns[-self.window:])

        returns_array = returns_array[np.isfinite(returns_array)]
        if returns_array.size < self.window * self.min_window_ratio:
            return

        finite_mask = np.isfinite(returns_array)
        if not finite_mask.all():
            returns_array = returns_array[finite_mask]
        if returns_array.size < self.window:
            return

        # check variance 
        returns_array -= returns_array.mean()
        var = np.var(returns_array, ddof=1)
        if var <= 1e-12:
            self._sigma = max(self._sigma, np.sqrt(var))
            return

        # fit GARCH model 
        model = arch_model(returns_array, mean='Zero', vol='GARCH', p=self.p, q=self.q, dist=self.dist, rescale=True)

        start_params = None 
        if self._last_fit is not None:
            try:
                start_params = self._last_fit.params.values
            except Exception: 
                start_params = None

        try: 
            fitted_model = model.fit(disp='off',
                                        show_warning=False,
                                        update_freq=0,
                                        options={'maxiter':150, 'ftol': 1e-4}
                                        )
            
            if not fitted_model.converged:
                raise RuntimeError("GARCH model failed to converge.")
            
            self._last_fit = fitted_model
            scaled_sigma = fitted_model.forecast(horizon=1, reindex=False).variance.values[-1,0]
            self._sigma = float(np.sqrt(scaled_sigma))
        
        except Exception as e:
            if self._last_fit is not None:
                forecast = self._last_fit.forecast(horizon=1, reindex=False)
                self._sigma = float(np.sqrt(forecast.variance.values[-1,0]))
            else: 
                self._sigma = float(np.std(returns_array))

    def predict(self) -> float:
        return self._sigma  



