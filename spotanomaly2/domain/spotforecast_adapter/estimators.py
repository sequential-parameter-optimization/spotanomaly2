"""Estimator wrappers + scaling registry for the spotforecast adapter.

- ``KernelRidgeApprox`` / ``SVRApprox``: Nyström-RBF + linear-on-top replacements
  for exact kernel KernelRidge / SVR that scale to full-year data.
- ``_MLPRegressorRobust``: ``MLPRegressor`` that drops zero-weight rows up front
  so sklearn's internal early-stopping validation split never sees an all-zero
  weight slice (it otherwise raises "Weights sum to zero, can't be normalized").
- ``_CatBoostRegressor``: subclass that swallows ``set_params(task_type='CPU')``
  errors that spotforecast2 issues against an already-fitted CatBoost model.
- ``_NEEDS_SCALING``: registry of model names that require StandardScaler-style
  ``transformer_y`` / ``transformer_exog`` rather than wrapping the estimator.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor as _SklearnMLPRegressor

try:
    from catboost import CatBoostRegressor as _CatBoostBase

    class _CatBoostRegressor(_CatBoostBase):
        """Suppress set_params errors on fitted models.

        spotforecast2 calls set_params(task_type='CPU') during prediction
        on an already-fitted model; CatBoost rejects this via its C++ backend.
        Defined at module level so joblib can resolve the class by import path
        when serializing fitted forecasters.
        """

        @property
        def task_type(self):
            return "CPU"

        def set_params(self, **params):
            try:
                return super().set_params(**params)
            except Exception:
                return self

except ImportError:
    _CatBoostRegressor = None  # type: ignore[assignment,misc]


class KernelRidgeApprox(BaseEstimator, RegressorMixin):
    """Nyström(rbf) → Ridge approximation of KernelRidge.

    Replaces the exact N×N Gram matrix with an ``N × n_components`` feature
    map, so fit cost drops from ``O(N³) / O(N²) memory`` to
    ``O(N·m + m³) / O(N·m) memory`` (where m = n_components). At m≈1500 the
    approximation is indistinguishable from the exact kernel for typical
    time-series data, but it scales to the full dataset — no kernel cap needed.
    """

    def __init__(
        self,
        n_components: int = 1500,
        gamma: float | None = None,
        alpha: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):  # noqa: N803 (sklearn convention)
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import Ridge

        self._nystroem = Nystroem(
            kernel="rbf",
            gamma=self.gamma,
            n_components=int(self.n_components),
            random_state=self.random_state,
        )
        x_t = self._nystroem.fit_transform(X)
        self._ridge = Ridge(alpha=float(self.alpha))
        if sample_weight is not None:
            self._ridge.fit(x_t, y, sample_weight=sample_weight)
        else:
            self._ridge.fit(x_t, y)
        return self

    def predict(self, X):  # noqa: N803 (sklearn convention)
        return self._ridge.predict(self._nystroem.transform(X))


class SVRApprox(BaseEstimator, RegressorMixin):
    """Nyström(rbf) → LinearSVR approximation of SVR.

    Same scaling story as ``KernelRidgeApprox``: finite-dim feature map +
    linear ε-insensitive regressor on top, so the model fits on full-year
    data without the N² memory blow-up of exact RBF-SVR.

    LinearSVR (older sklearn) does not accept ``sample_weight``; if a weight
    vector is given, zero-weight rows are dropped before fit so imputation
    masking still works.
    """

    def __init__(
        self,
        n_components: int = 1500,
        gamma: float | None = None,
        C: float = 1.0,  # noqa: N803 (sklearn convention)
        epsilon: float = 0.1,
        max_iter: int = 5000,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):  # noqa: N803 (sklearn convention)
        from sklearn.kernel_approximation import Nystroem
        from sklearn.svm import LinearSVR

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if not (sw > 0).all():
                mask = sw > 0
                X = X.iloc[mask] if hasattr(X, "iloc") else X[mask]  # noqa: N806
                y = y.iloc[mask] if hasattr(y, "iloc") else y[mask]

        self._nystroem = Nystroem(
            kernel="rbf",
            gamma=self.gamma,
            n_components=int(self.n_components),
            random_state=self.random_state,
        )
        x_t = self._nystroem.fit_transform(X)
        self._svr = LinearSVR(
            C=float(self.C),
            epsilon=float(self.epsilon),
            max_iter=int(self.max_iter),
            random_state=self.random_state,
        )
        self._svr.fit(x_t, y)
        return self

    def predict(self, X):  # noqa: N803 (sklearn convention)
        return self._svr.predict(self._nystroem.transform(X))


class _MLPRegressorRobust(_SklearnMLPRegressor):
    """``MLPRegressor`` that drops zero-weight rows before fit.

    sklearn's ``MLPRegressor`` splits an internal validation set when
    ``early_stopping=True``. If ``sample_weight`` zeros out a contiguous
    block of rows (e.g. an imputation-masked region) that lands inside the
    validation slice, sklearn raises *"Weights sum to zero, can't be
    normalized"* while normalising the validation loss. Dropping zero-weight
    rows up front sidesteps the issue and matches what ``SVRApprox`` does
    for ``LinearSVR``.
    """

    def fit(self, X, y, sample_weight=None):  # noqa: N803 (sklearn convention)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if not (sw > 0).all():
                mask = sw > 0
                X = X.iloc[mask] if hasattr(X, "iloc") else X[mask]  # noqa: N806
                y = y.iloc[mask] if hasattr(y, "iloc") else y[mask]
                sample_weight = sw[mask]
        return super().fit(X, y, sample_weight=sample_weight)


# Models that need standardised features. spotforecast2 applies the scaler
# itself via transformer_y / transformer_exog (see _create_forecaster), so we
# do NOT wrap the estimator — wrapping triggers the LinearModel-bypass bug in
# ForecasterRecursive._recursive_predict.
#
# Exact RBF SVR / KernelRidge are rejected — use KernelRidgeApprox / SVRApprox
# instead (the class docstrings explain why).
_NEEDS_SCALING = frozenset(
    {
        "ridge",
        "ridgeregressor",
        "elasticnet",
        "elasticnetregressor",
        "lasso",
        "lassoregressor",
        "bayesianridge",
        "bayesian_ridge",
        "huber",
        "huberregressor",
        "mlp",
        "mlpregressor",
        "kernelridgeapprox",
        "kernel_ridge_approx",
        "svrapprox",
        "svr_approx",
    }
)
