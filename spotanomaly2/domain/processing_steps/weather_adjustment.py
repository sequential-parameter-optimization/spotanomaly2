"""Subtract a weather temperature baseline from the temperature signal."""

from typing import Any, Optional

import pandas as pd

from spotanomaly2.domain.processing_steps.base import ProcessingStep


class WeatherAdjustmentStep(ProcessingStep):
    """Subtract the merged ``exogenous_weather_temperature_baseline`` from ``temperature``.

    Weather fetching and the rolling-baseline computation live in
    ``ExogenousWeatherJoiner`` (the joiner runs once before ``DataProcessor``).
    This step is a pure per-panel transform: if both ``temperature`` and the
    baseline column are present, subtract. The baseline is an intermediate, so it
    is dropped afterwards either way — it must not reach training/detection as an
    exogenous feature (or, worse, a forecast target).
    """

    name = "Adjusting temperature with weather baseline"
    baseline_col = "exogenous_weather_temperature_baseline"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        if self.baseline_col not in df.columns:
            return df
        df = df.copy()
        if "temperature" in df.columns:
            df["temperature"] = df["temperature"] - df[self.baseline_col]
            if self.logger:
                self.logger.info(f"Adjusted temperature for {len(df)} timestamps using merged baseline")
        df = df.drop(columns=[self.baseline_col])
        return df
