"""Partial Subsequence Matching (PSM) as a registry strategy.

Wraps the free functions in :mod:`.subsequence` so PSM is selectable by name
(``"psm"``) like every other :class:`~.base.ImputationMethod`. Single-value gaps
are filled with the neighbour mean first (PSM itself does not handle them), then
the PSM algorithm fills the remaining multi-point gaps.

Note: PSM requires a :class:`pandas.DatetimeIndex` with a set frequency; callers
are responsible for ensuring the series carries one (e.g. the process stage sets
the resample frequency before imputing).
"""

import pandas as pd

from spotanomaly2.domain.imputation.base import ImputationMethod
from spotanomaly2.domain.imputation.subsequence import fill_missing_with_mean, subsequence_imputation


class PSMImputation(ImputationMethod):
    """Single-gap mean fill followed by Partial Subsequence Matching."""

    def __init__(self, logger=None):
        """Initialize with an optional logger forwarded to the PSM algorithm.

        Args:
            logger: Optional logger; receives a warning per gap PSM cannot fill.
        """
        self.logger = logger

    def impute(self, series: pd.Series) -> pd.Series:
        """Fill single-value gaps with neighbour mean, then PSM the rest."""
        series = fill_missing_with_mean(series)
        return subsequence_imputation(series, logger=self.logger)
