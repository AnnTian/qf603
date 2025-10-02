import math
import pandas as pd
from .base import VolatilityModelBase

class EWMAVolModel(VolatilityModelBase):
    """
    Exponentially Weighted Moving Average (EWMA) volatility estimator.
    """

    def __init__(self, lam: float = 0.97):
        self.lam = lam
        self.prev_mid = None
        self.var = 0.0
        self._sigma = 0.0

    def update(self, ts: pd.Timestamp, mid: float, bid: float, ask: float):
        if self.prev_mid is None:
            self.prev_mid = mid
            return
        r = math.log((mid + 1e-12) / (self.prev_mid + 1e-12))
        self.var = self.lam * self.var + (1 - self.lam) * r * r
        self.prev_mid = mid
        self._sigma = math.sqrt(self.var)

    def predict(self) -> float:
        return self._sigma * self.prev_mid
