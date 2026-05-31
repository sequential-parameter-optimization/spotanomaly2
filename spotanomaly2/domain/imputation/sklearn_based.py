"""scikit-learn-backed imputation strategies (MICE, KNN).

These import scikit-learn lazily and degrade gracefully (``available = False``)
so the package stays importable without the optional dependency.
"""

import pandas as pd

from spotanomaly2.domain.imputation.base import ImputationMethod


class IterativeImputation(ImputationMethod):
    """MICE: Multiple Imputation by Chained Equations (scikit-learn based)."""

    def __init__(self, max_iter: int = 10):
        """Initialize with max iterations.

        Args:
            max_iter: Maximum iterations for IterativeImputer
        """
        self.max_iter = max_iter
        try:
            import sklearn.experimental.enable_iterative_imputer  # noqa: F401 - side-effect import for sklearn
            from sklearn.impute import IterativeImputer as SklearnIterativeImputer

            self.IterativeImputer = SklearnIterativeImputer
            self.available = True
        except ImportError:
            self.available = False
            self.IterativeImputer = None

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using sklearn IterativeImputer (MICE)."""
        if not self.available:
            raise ImportError("scikit-learn not installed. Cannot use IterativeImputation.")

        x = series.values.reshape(-1, 1)
        imputer = self.IterativeImputer(max_iter=self.max_iter, random_state=42)
        x_imputed = imputer.fit_transform(x)
        return pd.Series(x_imputed.flatten(), index=series.index)


class KNNSklearnImputation(ImputationMethod):
    """KNN imputation using scikit-learn."""

    def __init__(self, n_neighbors: int = 5):
        """Initialize with number of neighbors.

        Args:
            n_neighbors: Number of nearest neighbors
        """
        self.n_neighbors = n_neighbors
        try:
            from sklearn.impute import KNNImputer

            self.KNNImputer = KNNImputer
            self.available = True
        except ImportError:
            self.available = False
            self.KNNImputer = None

    def impute(self, series: pd.Series) -> pd.Series:
        """Impute using sklearn KNNImputer."""
        if not self.available:
            raise ImportError("scikit-learn not installed. Cannot use KNNSklearnImputation.")

        x = series.values.reshape(-1, 1)
        imputer = self.KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
        x_imputed = imputer.fit_transform(x)
        return pd.Series(x_imputed.flatten(), index=series.index)
