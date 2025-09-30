import pandas as pd
from abc import ABC, abstractmethod

class VolatilityModelBase(ABC):
    """
    Abstract base class for volatility models.
    All models must implement update() and predict().
    """

    @abstractmethod
    def update(self, ts: pd.Timestamp, mid: float, bid: float, ask: float) -> None:
        """Update model state based on new market snapshot."""
        pass

    @abstractmethod
    def predict(self) -> float:
        """Return the current volatility estimate (σ)."""
        pass