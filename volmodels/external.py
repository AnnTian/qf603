import pandas as pd
from typing import Callable, Optional
from .base import VolatilityModelBase

class ExternalVolModel(VolatilityModelBase):
    """
    Wraps an external volatility predictor (e.g. ML model, external service).
    """

    def __init__(self, predictor: Callable[[], float],
                 on_update: Optional[Callable[[pd.Timestamp, float, float, float], None]] = None):
        self._predictor = predictor
        self._on_update = on_update
        self._sigma = 0.0

    def update(self, ts: pd.Timestamp, mid: float, bid: float, ask: float):
        if self._on_update:
            self._on_update(ts, mid, bid, ask)
        self._sigma = float(self._predictor())

    def predict(self) -> float:
        return self._sigma
