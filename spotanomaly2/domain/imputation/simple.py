"""Simple imputation strategies: neighbour mean and forward/backward fill."""

import pandas as pd

from spotanomaly2.domain.imputation.base import ImputationMethod
from spotanomaly2.domain.imputation.subsequence import fill_missing_with_mean


class MeanNeighborImputation(ImputationMethod):
    """Fill with mean of neighboring values (CURRENT METHOD)."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Fill single missing values with mean of neighbors, then fall back to the series mean."""
        result = fill_missing_with_mean(series)
        return result.fillna(result.mean())


class ForwardFillImputation(ImputationMethod):
    """Forward fill: propagate last valid observation."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Forward fill missing values."""
        return series.ffill().bfill()


class BackwardFillImputation(ImputationMethod):
    """Backward fill: propagate next valid observation."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Backward fill missing values."""
        return series.bfill().ffill()
