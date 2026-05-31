"""Impute missing values using a configurable method."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from spotanomaly2.domain import imputation
from spotanomaly2.domain.processing.base import ProcessingStep


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

    name = "Imputing missing values"

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

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        # deep=True so the in-place index.freq assignment below cannot leak to the caller.
        df = df.copy()
        df.index = df.index.copy(deep=True)
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

        return self._impute(df)

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute every numeric column using the configured registry method.

        ``psm`` is a registered strategy like any other (see
        :class:`spotanomaly2.domain.imputation.PSMImputation`), so there is no
        special-casing here. If a method raises, fall back to PSM per column.
        """
        for col in df.columns:
            if col.endswith(self.weight_suffix):
                continue
            series = df[col].copy()
            try:
                # Skip non-numeric columns
                if series.dtype not in [np.float64, np.float32, np.int64, np.int32]:
                    continue

                # Apply imputation + shared weight flag creation
                imputed, observed_weight = imputation.impute_series_with_weight(
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
