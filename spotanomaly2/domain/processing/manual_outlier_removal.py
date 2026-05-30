"""Manually mark outliers as NaN based on configured thresholds."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from spotanomaly2.domain.processing.base import ProcessingStep


class ManualOutlierRemovalStep(ProcessingStep):
    """Manually mark outliers as NaN based on configured thresholds."""

    name = "Manual outlier removal"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
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

        if panel_id is None:
            return df

        panels_cfg = manual_cfg.get("panels", {})
        panel_cfg = panels_cfg.get(str(panel_id), {})
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
