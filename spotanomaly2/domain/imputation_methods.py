"""
Alternative imputation methods for time series missing value imputation.

This module provides several imputation strategies beyond simple mean imputation:
- Forward/Backward fill
- Interpolation (linear, spline)
- KNN-based methods
- Seasonal methods
- Rolling window methods
- Advanced methods (MICE, etc.)
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Scope suppression to known noisy warnings rather than silencing everything.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas")
warnings.filterwarnings("ignore", message=r".*ConvergenceWarning.*", module=r"sklearn")


class ImputationMethod:
    """Base class for imputation methods."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute missing values in a series.

        Args:
            series: Pandas Series with missing values (NaN)

        Returns:
            Series with imputed values
        """
        raise NotImplementedError


class MeanNeighborImputation(ImputationMethod):
    """Fill with mean of neighboring values (CURRENT METHOD)."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Fill single missing values with mean of neighbors (vectorized)."""
        result = series.copy()
        missing = result.isna()
        prev_valid = ~result.shift(1).isna()
        next_valid = ~result.shift(-1).isna()

        fillable = missing & prev_valid & next_valid
        if fillable.any():
            result[fillable] = (result.shift(1)[fillable] + result.shift(-1)[fillable]) / 2

        result = result.fillna(result.mean())
        return result


class ForwardFillImputation(ImputationMethod):
    """Forward fill: propagate last valid observation."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Forward fill missing values."""
        return series.fillna(method="ffill").fillna(method="bfill")


class BackwardFillImputation(ImputationMethod):
    """Backward fill: propagate next valid observation."""

    def impute(self, series: pd.Series) -> pd.Series:
        """Backward fill missing values."""
        return series.fillna(method="bfill").fillna(method="ffill")


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


class KNNTemporalImputation(ImputationMethod):
    """KNN imputation based on temporal proximity."""

    def __init__(self, n_neighbors: int = 5):
        """Initialize with number of neighbors.

        Args:
            n_neighbors: Number of nearest neighbors to use
        """
        self.n_neighbors = n_neighbors

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using temporal KNN."""
        result = series.copy()
        missing_indices = np.where(result.isna())[0]

        for idx in missing_indices:
            # Find k nearest non-missing values (temporally)
            distances = np.abs(np.arange(len(result)) - idx)
            valid_indices = np.where(~result.isna())[0]

            if len(valid_indices) == 0:
                continue

            # Get nearest neighbors
            distances_valid = distances[valid_indices]
            nearest_count = min(self.n_neighbors, len(valid_indices))
            nearest_indices = valid_indices[np.argsort(distances_valid)[:nearest_count]]

            # Weighted average: closer neighbors have higher weight
            neighbor_distances = np.abs(nearest_indices - idx).astype(float)
            weights = 1 / (neighbor_distances + 1)  # Avoid division by zero
            weights /= weights.sum()

            result.iloc[idx] = np.average(result.iloc[nearest_indices], weights=weights)

        return result


class SeasonalImputation(ImputationMethod):
    """Use seasonal patterns for imputation (assumes daily pattern).

    For water quality data with 5-minute intervals:
    - One day = 288 samples (24h × 12 per hour)
    """

    def __init__(self, period: int = 288):
        """Initialize with seasonal period.

        Args:
            period: Number of samples in one cycle (default 288 = 24h at 5min intervals)
        """
        self.period = period

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using seasonal patterns."""
        result = series.copy()
        missing_indices = np.where(result.isna())[0]

        for idx in missing_indices:
            seasonal_values = []

            # Look for same position in previous/next cycles
            for offset in [-self.period, self.period]:
                seasonal_idx = idx + offset
                if 0 <= seasonal_idx < len(result) and not pd.isna(result.iloc[seasonal_idx]):
                    seasonal_values.append(result.iloc[seasonal_idx])

            if seasonal_values:
                result.iloc[idx] = np.mean(seasonal_values)
            else:
                # Fallback to neighbor mean
                neighbors = []
                for offset in [-1, 1]:
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(result) and not pd.isna(result.iloc[neighbor_idx]):
                        neighbors.append(result.iloc[neighbor_idx])
                if neighbors:
                    result.iloc[idx] = np.mean(neighbors)

        return result.fillna(series.mean())


class RollingMeanImputation(ImputationMethod):
    """Fill using rolling mean from surrounding window."""

    def __init__(self, window: int = 10):
        """Initialize with window size.

        Args:
            window: Half-width of window around missing value
        """
        self.window = window

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using rolling window mean."""
        result = series.copy()
        missing_indices = np.where(result.isna())[0]

        for idx in missing_indices:
            start = max(0, idx - self.window)
            end = min(len(result), idx + self.window + 1)
            window_values = result.iloc[start:end].dropna()

            if len(window_values) > 0:
                result.iloc[idx] = window_values.mean()

        return result.fillna(series.mean())


class IterativeImputation(ImputationMethod):
    """MICE: Multiple Imputation by Chained Equations (scikit-learn based)."""

    def __init__(self, max_iter: int = 10):
        """Initialize with max iterations.

        Args:
            max_iter: Maximum iterations for IterativeImputer
        """
        self.max_iter = max_iter
        try:
            import sklearn.experimental.enable_iterative_imputer  # noqa: F401 - side-effect import for sklearn
            from sklearn.impute import IterativeImputer as SklearnIterativeImputer

            self.IterativeImputer = SklearnIterativeImputer
            self.available = True
        except ImportError:
            self.available = False
            self.IterativeImputer = None

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using sklearn IterativeImputer (MICE)."""
        if not self.available:
            raise ImportError("scikit-learn not installed. Cannot use IterativeImputation.")

        x = series.values.reshape(-1, 1)
        imputer = self.IterativeImputer(max_iter=self.max_iter, random_state=42)
        x_imputed = imputer.fit_transform(x)
        return pd.Series(x_imputed.flatten(), index=series.index)


class KNNSklearnImputation(ImputationMethod):
    """KNN imputation using scikit-learn."""

    def __init__(self, n_neighbors: int = 5):
        """Initialize with number of neighbors.

        Args:
            n_neighbors: Number of nearest neighbors
        """
        self.n_neighbors = n_neighbors
        try:
            from sklearn.impute import KNNImputer

            self.KNNImputer = KNNImputer
            self.available = True
        except ImportError:
            self.available = False
            self.KNNImputer = None

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using sklearn KNNImputer."""
        if not self.available:
            raise ImportError("scikit-learn not installed. Cannot use KNNSklearnImputation.")

        x = series.values.reshape(-1, 1)
        imputer = self.KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
        x_imputed = imputer.fit_transform(x)
        return pd.Series(x_imputed.flatten(), index=series.index)


# Registry of available methods
IMPUTATION_METHODS = {
    "mean": MeanNeighborImputation,
    "forward_fill": ForwardFillImputation,
    "backward_fill": BackwardFillImputation,
    "linear_interpolation": LinearInterpolationImputation,
    "spline_interpolation": SplineInterpolationImputation,
    "knn_temporal": KNNTemporalImputation,
    "seasonal": SeasonalImputation,
    "rolling_mean": RollingMeanImputation,
    "iterative": IterativeImputation,
    "knn_sklearn": KNNSklearnImputation,
}


def get_imputation_method(method_name: str, **kwargs) -> ImputationMethod:
    """Get an imputation method by name.

    Args:
        method_name: Name of the method (see IMPUTATION_METHODS keys)
        **kwargs: Additional arguments to pass to the method constructor

    Returns:
        ImputationMethod instance

    Raises:
        ValueError: If method_name is not recognized
    """
    if method_name not in IMPUTATION_METHODS:
        raise ValueError(
            f"Unknown imputation method: {method_name}. Available methods: {list(IMPUTATION_METHODS.keys())}"
        )

    return IMPUTATION_METHODS[method_name](**kwargs)


def impute_series(series: pd.Series, method: str = "linear_interpolation", **kwargs) -> pd.Series:
    """Convenience function to impute a series.

    Args:
        series: Pandas Series with missing values
        method: Imputation method name
        **kwargs: Additional arguments for the method

    Returns:
        Series with imputed values
    """
    imputer = get_imputation_method(method, **kwargs)
    return imputer.impute(series)


def impute_series_with_weight(
    series: pd.Series,
    method: str = "linear_interpolation",
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """Impute a series and return matching observed/imputed weight flag.

    Weight semantics:
    - 1: original value was observed (non-NaN)
    - 0: value was missing and therefore imputed
    """
    observed_weight = (~series.isna()).astype(int)
    imputed = impute_series(series, method=method, **kwargs)
    return imputed, observed_weight


def impute_dataframe(
    df: pd.DataFrame, method: str = "linear_interpolation", columns: Optional[list] = None, **kwargs
) -> pd.DataFrame:
    """Impute a DataFrame column by column.

    Args:
        df: DataFrame with missing values
        method: Imputation method name
        columns: Specific columns to impute (default: all numeric)
        **kwargs: Additional arguments for the method

    Returns:
        DataFrame with imputed values
    """
    result = df.copy()

    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in result.columns:
            result[col] = impute_series(result[col], method=method, **kwargs)

    return result
