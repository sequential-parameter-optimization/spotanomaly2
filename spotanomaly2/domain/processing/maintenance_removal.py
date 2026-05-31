"""Remove maintenance periods from data."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from spotanomaly2.domain.processing.base import ProcessingStep


class MaintenanceRemovalStep(ProcessingStep):
    """Remove maintenance periods from data (set to NaN, then drop flag column)."""

    name = "Removing maintenance periods"

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger

    def process(self, df: pd.DataFrame, panel_id: Optional[str] = None) -> pd.DataFrame:
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
