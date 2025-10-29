from arch import arch_model
import math
import numpy as np 
import pandas as pd 
from .base import VolatilityModelBase

#TODO: check if we should implement multivariate garch model (for highly correlated assets)
class GARCHVolModel(VolatilityModelBase):
    """
    GARCH volatility model for real-time volatility estimation.
    Uses a rolling window of returns to fit GARCH(3,1) model.
    """
    
    def __init__(self, window: int = 1000, p: int = 3, q: int = 1, refit_freq: int = 20, dist: str = 't'):
        """
        Args:
            window: number of past returns to use for GARCH(1,1) fitting
            p: number of ARCH terms
            q: number of GARCH terms
            refit_freq: frequency of model refitting 
            dist: t distribution for residuals (heavy tails)
        """
        self.window = window
        self.p = p
        self.q = q
        self.refit_freq = refit_freq
        self.dist = dist 
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
        if returns_array.size < self.window * 0.8:
            return

        finite_mask = np.isfinite(returns_array)
        if not finite_mask.all():
            returns_array = returns_array[finite_mask]
        if returns_array.size < self.window:
            # not enough valid data to refit
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

    # TODO: need to check on this logic  
    def predict(self) -> float:
        return self._sigma  


    #TODO: split into train and test 80-20? then converge on ewma 

