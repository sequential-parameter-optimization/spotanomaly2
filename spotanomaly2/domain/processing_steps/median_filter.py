"""Apply median filter to flow columns to remove spikes."""

from typing import Any, Optional

import pandas as pd
import scipy.signal

from spotanomaly2.domain.processing_steps.base import ProcessingStep


class MedianFilterStep(ProcessingStep):
    """Apply median filter to flow columns to remove spikes."""

    name = "Applying median filter to flow columns"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
        flow_pattern = self.config["process"]["flow_columns_pattern"]
        kernel_size = self.config["process"]["median_filter_kernel"]
        for col in df.columns:
            if col.endswith(self.weight_suffix):
                continue
            if flow_pattern in col:
                filtered = scipy.signal.medfilt(df[col].values, kernel_size=kernel_size)
                df[col] = pd.Series(filtered, index=df.index, name=col)
        return df
