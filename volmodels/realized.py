import math
import pandas as pd
from collections import deque
from .base import VolatilityModelBase

class RealizedVolModel(VolatilityModelBase):
    """
    Realized volatility estimator using a rolling window of log returns.
    Computes the square root of the average squared returns over the window.
    """

    def __init__(self, window: int = 50):
        """
        Args:
            window: number of past returns to use for realized volatility.
        """
        self.window = window
        self.prev_mid = None
        self.returns = deque(maxlen=window)
        self._sigma = 0.0

    def update(self, ts: pd.Timestamp, mid: float, bid: float, ask: float):
        if self.prev_mid is None:
            self.prev_mid = mid
            return
        r = math.log((mid + 1e-12) / (self.prev_mid + 1e-12))
        self.returns.append(r)
        self.prev_mid = mid

        if len(self.returns) > 0:
            mean_sq = sum(r*r for r in self.returns) / len(self.returns)
            self._sigma = math.sqrt(mean_sq)

    def predict(self) -> float:
        return self._sigma
