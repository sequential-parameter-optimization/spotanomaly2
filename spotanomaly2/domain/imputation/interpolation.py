"""Interpolation-based imputation strategies."""

import pandas as pd

from spotanomaly2.domain.imputation.base import ImputationMethod


class LinearInterpolationImputation(ImputationMethod):
    """Linear interpolation between valid values."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Interpolate linearly."""
        return series.interpolate(method="linear", limit_direction="both").fillna(series.mean())


class SplineInterpolationImputation(ImputationMethod):
    """Spline interpolation: smooth polynomial fit."""

    def __init__(self, order: int = 3):
        """Initialize with spline order.

        Args:
            order: Polynomial order for spline (default 3 = cubic)
        """
        self.order = order

    def impute(self, series: pd.Series) -> pd.Series:
        """Interpolate using spline."""
        try:
            result = series.interpolate(method="spline", order=self.order, limit_direction="both")
        except (ValueError, Exception):
            # Fallback to linear if spline fails
            result = series.interpolate(method="linear", limit_direction="both")
        return result.fillna(series.mean())
