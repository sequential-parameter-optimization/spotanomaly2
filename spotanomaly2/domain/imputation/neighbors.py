"""Neighbour- and pattern-based imputation strategies (no external deps)."""

import numpy as np
import pandas as pd

from spotanomaly2.domain.imputation.base import ImputationMethod


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
