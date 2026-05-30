"""Base class for the imputation strategy registry."""

import pandas as pd


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
