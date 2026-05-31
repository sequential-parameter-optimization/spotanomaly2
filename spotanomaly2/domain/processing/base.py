"""Abstract base class for processing steps."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class ProcessingStep(ABC):
    """Base class for a single processing step."""

    name: str = ""

    @abstractmethod
    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        """Apply this step to the DataFrame. Returns modified DataFrame.

        ``panel_id`` is accepted by every step's signature but only consumed by
        steps whose behavior depends on it (e.g. ``ManualOutlierRemovalStep``).
        """
        pass
