"""Focused processing steps for the data pipeline (Strategy-style steps)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.signal

from spotanomaly2.domain import imputation, imputation_methods
from spotanomaly2.domain.weather_fetcher import WeatherFetcher


class ProcessingStep(ABC):
    """Base class for a single processing step."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply this step to the DataFrame. Returns modified DataFrame."""
        pass


class ResampleStep(ProcessingStep):
    """Resample DataFrame to target frequency."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        resample_cfg = self.config["process"]["resample"]
        return df.resample(
            rule=resample_cfg["freq"],
            origin=resample_cfg["origin"],
            label=resample_cfg["label"],
            closed=resample_cfg["closed"],
        ).mean()


class MaintenanceRemovalStep(ProcessingStep):
    """Remove maintenance periods from data (set to NaN, then drop flag column)."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        maint_col = self.config["process"]["maintenance_column"]
        if maint_col not in df.columns:
            if self.logger:
                self.logger.warning(f"Maintenance column '{maint_col}' not found in data")
            return df
        maint_flag = df[maint_col].astype(bool)
        df = df.copy()
        df.loc[maint_flag] = np.nan
        df.drop(columns=[maint_col], inplace=True)
        return df


class ManualOutlierRemovalStep(ProcessingStep):
    """Manually mark outliers as NaN based on configured thresholds."""

    def __init__(self, config: dict[str, Any], logger=None, panel_id: Optional[str] = None):
        self.config = config
        self.logger = logger
        self.panel_id = panel_id

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual outlier removal based on configuration.

        Expects configuration under process.manual_outliers:
          enabled: bool
          panels:
            <panel_id>:
              columns:
                <column_name>:
                  lower: <float|null>
                  upper: <float|null>
        """
        manual_cfg = self.config.get("process", {}).get("manual_outliers", {})
        if not manual_cfg.get("enabled", False):
            return df

        panels_cfg = manual_cfg.get("panels", {})
        if self.panel_id is None:
            return df

        panel_cfg = panels_cfg.get(str(self.panel_id), {})
        columns_cfg = panel_cfg.get("columns", {})
        if not columns_cfg:
            return df

        df = df.copy()
        for col, thresholds in columns_cfg.items():
            if col not in df.columns:
                if self.logger:
                    self.logger.warning(f"Manual outlier removal configured for '{col}' but column not found")
                continue

            lower = thresholds.get("lower")
            upper = thresholds.get("upper")

            if lower is None and upper is None:
                continue

            if lower is not None and upper is not None:
                mask = (df[col] > upper) | (df[col] < lower)
            elif lower is not None:
                mask = df[col] < lower
            else:
                mask = df[col] > upper

            n_outliers = int(mask.sum())
            if n_outliers > 0:
                df.loc[mask, col] = np.nan
                if self.logger:
                    self.logger.info(f"Manual outlier removal: marked {n_outliers} values in '{col}'")

        return df


class ImputationStep(ProcessingStep):
    """Impute missing values using configurable method.

    Supports multiple imputation strategies:
    - 'psm': Partial Subsequence Matching (legacy method)
    - 'mean': Mean of neighbors
    - 'forward_fill': Forward fill
    - 'linear_interpolation': Linear interpolation
    - 'spline_interpolation': Spline interpolation
    - 'knn_temporal': KNN based on temporal proximity
    - 'seasonal': Using daily patterns
    - 'rolling_mean': Rolling window mean
    """

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

        # Get imputation method from config, default to linear interpolation
        self.method = self.config.get("process", {}).get("imputation", {}).get("method", "linear_interpolation")

        # Get method-specific parameters
        self.method_params = self.config.get("process", {}).get("imputation", {}).get("params", {})

        imputation_cfg = self.config.get("process", {}).get("imputation", {})
        self.add_weights = imputation_cfg.get("add_weights", True)
        self.weight_suffix = imputation_cfg.get("weight_suffix", "__weight")

        if self.logger:
            self.logger.info(f"Imputation method: {self.method}")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Set frequency on index if not already set
        if df.index.freq is None:
            freq_str = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
            try:
                if len(df) >= 3:
                    inferred = pd.infer_freq(df.index)
                    if inferred:
                        df.index.freq = inferred
                if df.index.freq is None:
                    df.index.freq = pd.tseries.frequencies.to_offset(freq_str)
            except (ValueError, TypeError):
                df.index.freq = pd.tseries.frequencies.to_offset(freq_str)

        if self.method == "psm":
            # Use original PSM + mean imputation (current behavior)
            return self._impute_psm(df)
        else:
            # Use new imputation methods
            return self._impute_alternative(df)

    def _impute_psm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Original PSM-based imputation (backward compatible)."""
        for col in df.columns:
            if col.endswith(self.weight_suffix):
                continue
            series = df[col].copy()
            imputed_mask = series.isna()
            series = imputation.fill_missing_with_mean(series)
            series = imputation.subsequence_imputation(series)
            df[col] = series
            if self.add_weights and series.dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[f"{col}{self.weight_suffix}"] = (~imputed_mask).astype(int)
        return df

    def _impute_alternative(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use alternative imputation methods."""
        for col in df.columns:
            if col.endswith(self.weight_suffix):
                continue
            series = df[col].copy()
            try:
                # Skip non-numeric columns
                if series.dtype not in [np.float64, np.float32, np.int64, np.int32]:
                    continue

                # Apply imputation + shared weight flag creation
                imputed, observed_weight = imputation_methods.impute_series_with_weight(
                    series, method=self.method, **self.method_params
                )
                df[col] = imputed

                if self.add_weights:
                    df[f"{col}{self.weight_suffix}"] = observed_weight

                if self.logger:
                    self.logger.debug(f"Imputed column {col} using {self.method}")

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to impute {col}: {e}. Using fallback (mean).")
                # Fallback to mean imputation
                series = imputation.fill_missing_with_mean(series)
                series = imputation.subsequence_imputation(series)
                df[col] = series

        return df


class MedianFilterStep(ProcessingStep):
    """Apply median filter to flow columns to remove spikes."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        flow_pattern = self.config["process"]["flow_columns_pattern"]
        kernel_size = self.config["process"]["median_filter_kernel"]
        for col in df.columns:
            if col.endswith(self.weight_suffix):
                continue
            if flow_pattern in col:
                filtered = scipy.signal.medfilt(df[col].values, kernel_size=kernel_size)
                df[col] = pd.Series(filtered, index=df.index, name=col)
        return df


class TemperatureAggregationStep(ProcessingStep):
    """Average all temperature sensors into a single 'temperature' column."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        temp_pattern = self.config["process"]["temperature_columns_pattern"]
        temp_cols = [col for col in df.columns if temp_pattern in col and not col.endswith(self.weight_suffix)]
        if not temp_cols:
            if self.logger:
                self.logger.warning("No temperature columns found to aggregate")
            return df
        df = df.copy()
        df["temperature"] = df[temp_cols].mean(axis=1)
        df.drop(columns=temp_cols, inplace=True)
        return df


class WeatherAdjustmentStep(ProcessingStep):
    """Adjust temperature by subtracting rolling weather baseline."""

    def __init__(
        self,
        config: dict[str, Any],
        logger=None,
        weather_fetcher: Optional[WeatherFetcher] = None,
    ):
        self.config = config
        self.logger = logger
        self.weather_fetcher = weather_fetcher

    def _fetch_weather_baseline(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.weather_fetcher is None or len(df) == 0:
            return None
        weather_cfg = self.config.get("process", {}).get("weather", {})
        lookback_days = weather_cfg.get("lookback_days", 14)
        panel_start = df.index.min()
        panel_end = df.index.max()
        weather_start = panel_start - pd.Timedelta(days=lookback_days)
        try:
            if self.logger:
                self.logger.info(f"Fetching weather data for baseline: {weather_start} to {panel_end}")
            fallback_on_failure = weather_cfg.get("fallback_on_failure", True)
            cache_path_str = weather_cfg.get("cache_path")
            cache_path = Path(cache_path_str) if cache_path_str else None
            weather_df = self.weather_fetcher.get_weather_data(
                start=weather_start,
                end=panel_end,
                cache_path=cache_path,
                timezone="UTC",
                fallback_on_failure=fallback_on_failure,
            )
            if "temperature_2m" not in weather_df.columns:
                return None
            return weather_df
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to fetch weather baseline: {e}")
            return None

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if "temperature" not in df.columns or self.weather_fetcher is None:
            return df
        weather_df = self._fetch_weather_baseline(df)
        if weather_df is None:
            if self.logger:
                self.logger.warning("Could not fetch weather baseline, skipping adjustment")
            return df
        weather_cfg = self.config.get("process", {}).get("weather", {})
        lookback_days = weather_cfg.get("lookback_days", 14)
        panel_freq = self.config["process"]["resample"]["freq"]
        weather_resampled = weather_df["temperature_2m"].resample(panel_freq).ffill()
        # IMPORTANT: compute the rolling mean on the full dense resampled series BEFORE
        # aligning to df.index. Aligning first produces a sparse series; rolling over a
        # sparse series yields a mean of only the few present timestamps instead of the
        # true 14-day window — this is exactly what caused the cutoff-aligned temperature
        # spike when live chunks are small (e.g. a few rows at the end of a run).
        weather_baseline_full = weather_resampled.rolling(window=f"{lookback_days}D", min_periods=1).mean()
        if weather_baseline_full.isna().any():
            weather_baseline_full = weather_baseline_full.fillna(weather_resampled.mean())
        weather_baseline = weather_baseline_full.reindex(df.index, method="ffill")
        df = df.copy()
        df["temperature"] = df["temperature"] - weather_baseline
        if self.logger:
            self.logger.info(f"Adjusted temperature for {len(df)} timestamps (window: {lookback_days} days)")

        # Append configured weather feature columns (e.g. snow_depth) to the DataFrame
        feature_columns = weather_cfg.get("feature_columns", [])
        for col in feature_columns:
            if col in weather_df.columns:
                resampled = weather_df[col].resample(panel_freq).ffill()
                df[f"weather_{col}"] = resampled.reindex(df.index, method="ffill")
                if self.logger:
                    self.logger.info(f"Added weather feature column: weather_{col}")
            else:
                if self.logger:
                    self.logger.warning(f"Configured weather feature column '{col}' not available in weather data")

        return df
