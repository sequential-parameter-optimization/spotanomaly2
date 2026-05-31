"""Registry of imputation strategies and convenience entry points."""

from typing import Optional

import numpy as np
import pandas as pd

from spotanomaly2.domain.imputation.base import ImputationMethod
from spotanomaly2.domain.imputation.interpolation import (
    LinearInterpolationImputation,
    SplineInterpolationImputation,
)
from spotanomaly2.domain.imputation.neighbors import (
    KNNTemporalImputation,
    RollingMeanImputation,
    SeasonalImputation,
)
from spotanomaly2.domain.imputation.psm import PSMImputation
from spotanomaly2.domain.imputation.simple import (
    BackwardFillImputation,
    ForwardFillImputation,
    MeanNeighborImputation,
)
from spotanomaly2.domain.imputation.sklearn_based import (
    IterativeImputation,
    KNNSklearnImputation,
)

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
    "psm": PSMImputation,
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
