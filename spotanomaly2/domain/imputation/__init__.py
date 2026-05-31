"""Imputation strategies for filling missing values in time-series data.

This package unifies two previously separate concerns:

- **PSM internals** (:mod:`.subsequence`) — gap detection and Partial Subsequence
  Matching, exposed both as free functions and as the ``"psm"`` strategy.
- **Strategy registry** (:mod:`.registry`) — a name-keyed registry of
  :class:`~.base.ImputationMethod` implementations (mean, fills, interpolation,
  KNN, seasonal, rolling, sklearn-backed, and PSM).

The public names below are re-exported for backwards compatibility with the old
``spotanomaly2.domain.imputation`` and ``spotanomaly2.domain.imputation_methods``
modules.
"""

import warnings

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
from spotanomaly2.domain.imputation.registry import (
    IMPUTATION_METHODS,
    get_imputation_method,
    impute_dataframe,
    impute_series,
    impute_series_with_weight,
)
from spotanomaly2.domain.imputation.simple import (
    BackwardFillImputation,
    ForwardFillImputation,
    MeanNeighborImputation,
)
from spotanomaly2.domain.imputation.sklearn_based import (
    IterativeImputation,
    KNNSklearnImputation,
)
from spotanomaly2.domain.imputation.subsequence import (
    fill_missing_with_mean,
    identify_missing_data_gaps_with_count,
    series_mean,
    subsequence_imputation,
    vectorized_subsequence_distances,
)

# Scope suppression to known noisy warnings rather than silencing everything.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas")
warnings.filterwarnings("ignore", message=r".*ConvergenceWarning.*", module=r"sklearn")

__all__ = [
    # PSM internals
    "identify_missing_data_gaps_with_count",
    "vectorized_subsequence_distances",
    "series_mean",
    "fill_missing_with_mean",
    "subsequence_imputation",
    # Strategy base + registry
    "ImputationMethod",
    "IMPUTATION_METHODS",
    "get_imputation_method",
    "impute_series",
    "impute_series_with_weight",
    "impute_dataframe",
    # Concrete strategies
    "MeanNeighborImputation",
    "ForwardFillImputation",
    "BackwardFillImputation",
    "LinearInterpolationImputation",
    "SplineInterpolationImputation",
    "KNNTemporalImputation",
    "SeasonalImputation",
    "RollingMeanImputation",
    "IterativeImputation",
    "KNNSklearnImputation",
    "PSMImputation",
]
