"""Adapter wrapping spotforecast2's ForecasterRecursive for spotanomaly2's pipeline.

spotanomaly2's own data-processing pipeline (5-minute resampling, imputation, ...)
runs *before* this adapter, so we go straight to ``ForecasterRecursive`` instead of
through ``MultiTask.prepare_data()`` and add only what's domain-specific:

- per-channel model selection from YAML
- training-sample weighting that respects spotanomaly2's imputation flags
- a cheap Ridge pre-pass to mask anomalous training regions
- batch one-step-ahead prediction over an arbitrary held-out frame

Feature scaling for non-tree models is delegated to spotforecast2's built-in
``transformer_y`` / ``transformer_exog`` rather than wrapping the estimator.
"""

import inspect
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor as _SklearnMLPRegressor

from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp

try:
    from catboost import CatBoostRegressor as _CatBoostBase

    class _CatBoostRegressor(_CatBoostBase):
        """Suppress set_params errors on fitted models.

        spotforecast2 calls set_params(task_type='CPU') during prediction
        on an already-fitted model; CatBoost rejects this via its C++ backend.
        Defined at module level so joblib/pickle can resolve the class path.
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


def _time_series_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _mask_known_anomalies(
    df: pd.DataFrame,
    known_anomalies: list[dict],
    buffer: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Set selected columns to NaN within known anomaly windows (extended by buffer)."""
    if not known_anomalies:
        return df
    buffer_td = pd.Timedelta(buffer)
    df = df.copy()
    cols_to_mask = columns if columns is not None else list(df.columns)
    for anomaly in known_anomalies:
        start_str = anomaly.get("start")
        end_str = anomaly.get("end")
        if not start_str or not end_str:
            continue
        try:
            a_start = pd.to_datetime(start_str)
            a_end = pd.to_datetime(end_str)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                a_start = a_start.tz_convert(df.index.tz) if a_start.tz else a_start.tz_localize(df.index.tz)
                a_end = a_end.tz_convert(df.index.tz) if a_end.tz else a_end.tz_localize(df.index.tz)
            mask = (df.index >= a_start - buffer_td) & (df.index <= a_end + buffer_td)
            existing_cols = [c for c in cols_to_mask if c in df.columns]
            if existing_cols:
                df.loc[mask, existing_cols] = np.nan
        except (ValueError, TypeError):
            continue
    return df


def _get_weight_suffix_from_config(config: dict[str, Any]) -> str:
    """Return configured imputation weight suffix."""
    return config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")


def _resolve_feature_columns(
    config: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Resolve (target_cols, exog_columns) consistently for train/predict/tune.

    Derives exogenous and target columns from config so the tuner sees the same
    model topology as the training step: ``exogenous_*`` columns are fed as exog
    features (unless ``weight_residuals`` consumes them), and they are NOT scored
    as targets.
    """
    configured_exog_columns = config.get("train", {}).get("exog_columns", [])
    weight_residuals_enabled = config.get("exogenous", {}).get("weight_residuals", {}).get("enabled", False)
    weight_suffix = _get_weight_suffix_from_config(config)

    if weight_residuals_enabled:
        prefixed_exog_columns: list[str] = []
    else:
        prefixed_exog_columns = [
            col for col in df.columns if col.startswith("exogenous_") and not col.endswith(weight_suffix)
        ]
    exog_columns = list(
        dict.fromkeys([col for col in [*configured_exog_columns, *prefixed_exog_columns] if col in df.columns])
    )
    target_cols = [
        col
        for col in df.columns
        if not col.endswith(weight_suffix) and col not in exog_columns and not col.startswith("exogenous_")
    ]
    return target_cols, exog_columns


def _compute_observed_mask(df: pd.DataFrame, target_col: str, weight_suffix: str) -> pd.Series:
    """Return True for observed (non-imputed) rows of a target column."""
    weight_col = f"{target_col}{weight_suffix}"
    if weight_col in df.columns:
        return df[weight_col].fillna(0.0) >= 0.5
    return pd.Series(True, index=df.index)


def _build_strict_training_sample_mask(observed_mask: pd.Series, n_lags: int) -> pd.Series:
    """Return True only when target and all required lag inputs are observed."""
    mask = observed_mask.fillna(False).astype(bool).copy()
    for lag in range(1, n_lags + 1):
        mask &= observed_mask.shift(lag).fillna(False).astype(bool)
    return mask


def _detect_anomalies_via_ridge(
    y: pd.Series,
    n_lags: int,
    threshold_scale: float = 4.0,
    buffer: int = 3,
) -> pd.Series:
    """Cheap Ridge-residual pre-pass to flag anomalous training regions."""
    from sklearn.linear_model import Ridge

    clean_mask = pd.Series(False, index=y.index)
    n = len(y)
    lag = min(n_lags, 24)
    if n < lag + 20:
        return clean_mask

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    indices: list[Any] = []
    vals = y.to_numpy(dtype=float)
    for t in range(lag, n):
        if np.any(np.isnan(vals[t - lag : t + 1])):
            continue
        x_rows.append(vals[t - lag : t][::-1])
        y_rows.append(vals[t])
        indices.append(y.index[t])

    if len(x_rows) < lag + 10:
        return clean_mask

    x_mat = np.array(x_rows)
    y_arr = np.array(y_rows)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_mat, y_arr)
    preds = ridge.predict(x_mat)
    resid = np.abs(y_arr - preds)

    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) or 1e-12
    threshold = med + threshold_scale * mad

    flagged_indices = {indices[i] for i, f in enumerate(resid > threshold) if f}

    all_idx = list(y.index)
    expanded: set = set()
    for idx in flagged_indices:
        try:
            pos = all_idx.index(idx)
        except ValueError:
            continue
        for offset in range(-buffer, buffer + 1):
            p = pos + offset
            if 0 <= p < n:
                expanded.add(all_idx[p])

    clean_mask.loc[list(expanded)] = True
    return clean_mask


def _ensure_datetime_freq(obj, fallback_freq: str | None = None):
    """Set ``obj.index.freq`` from ``pd.infer_freq``, falling back to
    ``fallback_freq`` when inference fails. Works on Series and DataFrame.
    """
    if isinstance(obj.index, pd.DatetimeIndex) and obj.index.freq is None:
        freq = pd.infer_freq(obj.index) or fallback_freq
        if freq:
            obj = obj.asfreq(freq)
    return obj


def _interpolate_for_model(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Interpolate and fill missing values in a Series or DataFrame for model consumption."""
    if isinstance(data, pd.Series):
        if not data.isna().any():
            return data.copy()
    elif not data.isna().any().any():
        return data
    data = data.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        return data.interpolate(method="time").bfill().ffill()
    return data.interpolate(method="linear").bfill().ffill()


def _difference(y: pd.Series, order: int) -> pd.Series:
    """Return ``y.diff()`` applied ``order`` times. Order 0 is a passthrough.

    Differencing makes the target stationary; a degenerate predictor on
    Δy then maps to the lag-1 baseline after one-step integration instead
    of to the training mean.
    """
    if order <= 0:
        return y
    out = y
    for _ in range(order):
        out = out.diff()
    return out


def _integrate_one_step(diff_preds: pd.Series, raw_y: pd.Series, order: int) -> pd.Series:
    """Integrate one-step-ahead Δy predictions back to raw scale.

    Anchored to the *actual* ``y[t-1]`` (not the previous prediction), so
    it stays stable over any horizon. Only order 1 is supported.
    """
    if order <= 0:
        return diff_preds
    if order == 1:
        return diff_preds + raw_y.shift(1).reindex(diff_preds.index)
    raise ValueError(f"Only differentiation order 1 is supported (got {order})")


def _one_step_ahead_predict(
    forecaster,
    full_y: pd.Series,
    full_exog: pd.DataFrame | None,
    target_index: pd.Index,
) -> np.ndarray:
    """One-step-ahead predictions for ``target_index``, using real observed lags.

    Builds lag features via ``forecaster.create_train_X_y``, calls the bare
    estimator (skipping ``ForecasterRecursive``'s recursive predict path),
    and inverse-scales when ``transformer_y`` is set. The recursive path
    cascade-diverges over thousands of steps for non-tree models; this
    one-step path is what detection / scoring actually uses, so train-time
    eval should report it too.
    """
    from spotforecast2_safe.forecaster.utils import transform_numpy

    full_y = _ensure_datetime_freq(full_y)
    if full_exog is not None:
        full_exog = _ensure_datetime_freq(full_exog)

    x_features, y_target = forecaster.create_train_X_y(y=full_y, exog=full_exog)
    preds_all = forecaster.estimator.predict(x_features)
    if forecaster.transformer_y is not None:
        preds_all = transform_numpy(
            array=np.asarray(preds_all, dtype=float),
            transformer=forecaster.transformer_y,
            fit=False,
            inverse_transform=True,
        )
    preds_series = pd.Series(preds_all, index=y_target.index)
    return preds_series.reindex(target_index).to_numpy()


def _predict_one_step_integrated(
    forecaster,
    full_y_raw: pd.Series,
    full_exog: pd.DataFrame | None,
    target_index: pd.Index,
    diff_order: int,
) -> np.ndarray:
    """One-step-ahead prediction with optional Δy → raw integration.

    For ``diff_order > 0`` the forecaster was fit on differenced ``y``; we
    rebuild Δy features, predict, then integrate one step from the real
    ``y[t-1]`` in ``full_y_raw``. This is the path production / detection
    takes — shared between train-time eval and ``SpotforecastTrainer.predict``.
    """
    if diff_order <= 0:
        return _one_step_ahead_predict(forecaster, full_y_raw, full_exog, target_index)
    y_for_features = _difference(full_y_raw, diff_order).dropna()
    exog_for_features = full_exog.loc[y_for_features.index] if full_exog is not None else None
    diff_preds_arr = _one_step_ahead_predict(forecaster, y_for_features, exog_for_features, y_for_features.index)
    diff_preds = pd.Series(diff_preds_arr, index=y_for_features.index)
    return _integrate_one_step(diff_preds, full_y_raw, diff_order).reindex(target_index).to_numpy()


def _filter_params(estimator_cls, raw_params: dict[str, Any], logger=None, model_name: str = "") -> dict[str, Any]:
    """Drop kwargs that ``estimator_cls.__init__`` does not accept."""
    sig = inspect.signature(estimator_cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return raw_params
    valid_keys = {key for key in sig.parameters if key != "self"}
    filtered = {k: v for k, v in raw_params.items() if k in valid_keys}
    dropped = sorted(k for k in raw_params if k not in valid_keys)
    if dropped and logger is not None:
        logger.warning(f"Ignoring unsupported params for {model_name}: {dropped}")
    return filtered


def _build_estimator(
    model_name: str,
    model_params: dict[str, Any] | None,
    random_seed: int,
    logger=None,
):
    """Map a model name to a fresh, unfitted sklearn-compatible estimator."""
    name = (model_name or "").strip().lower()
    params = dict(model_params or {})

    def _filt(cls):
        return _filter_params(cls, params, logger=logger, model_name=model_name)

    if name in {"lightgbm", "lgbm", "lgbmregressor"}:
        from lightgbm import LGBMRegressor

        params.setdefault("random_state", random_seed)
        params.setdefault("verbose", -1)
        return LGBMRegressor(**_filt(LGBMRegressor))

    if name in {"xgboost", "xgb", "xgbregressor"}:
        from xgboost import XGBRegressor

        params.setdefault("random_state", random_seed)
        return XGBRegressor(**_filt(XGBRegressor))

    if name in {"catboost", "catboostregressor"}:
        if _CatBoostRegressor is None:
            raise ImportError("catboost is not installed")
        params.setdefault("random_seed", random_seed)
        params.setdefault("verbose", 0)
        return _CatBoostRegressor(**_filt(_CatBoostRegressor))

    if name in {"ridge", "ridgeregressor"}:
        from sklearn.linear_model import Ridge

        return Ridge(**_filt(Ridge))

    if name in {"elasticnet", "elasticnetregressor"}:
        from sklearn.linear_model import ElasticNet

        params.setdefault("random_state", random_seed)
        return ElasticNet(**_filt(ElasticNet))

    if name in {"lasso", "lassoregressor"}:
        from sklearn.linear_model import Lasso

        params.setdefault("random_state", random_seed)
        return Lasso(**_filt(Lasso))

    if name in {"bayesianridge", "bayesian_ridge"}:
        from sklearn.linear_model import BayesianRidge

        return BayesianRidge(**_filt(BayesianRidge))

    if name in {"huber", "huberregressor"}:
        from sklearn.linear_model import HuberRegressor

        return HuberRegressor(**_filt(HuberRegressor))

    if name in {"kernelridgeapprox", "kernel_ridge_approx", "nystroemkernelridge"}:
        params.setdefault("random_state", random_seed)
        return KernelRidgeApprox(**_filt(KernelRidgeApprox))

    if name in {"svrapprox", "svr_approx", "nystroemsvr"}:
        params.setdefault("random_state", random_seed)
        return SVRApprox(**_filt(SVRApprox))

    if name in {"mlp", "mlpregressor"}:
        params.setdefault("random_state", random_seed)
        params.setdefault("max_iter", 500)
        params.setdefault("early_stopping", True)
        return _MLPRegressorRobust(**_filt(_MLPRegressorRobust))

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported: LightGBM, XGBoost, CatBoost, "
        "Ridge, ElasticNet, Lasso, BayesianRidge, Huber, KernelRidgeApprox, SVRApprox, MLP"
    )


def _create_forecaster(
    model_name: str,
    model_params: dict[str, Any] | None,
    n_lags: int | list[int],
    *,
    random_seed: int = 42,
    has_exog: bool = False,
    logger=None,
):
    """Create a ``ForecasterRecursive`` configured for ``model_name``.

    For scale-sensitive models (linear / kernel / MLP) we pass spotforecast2's
    ``transformer_y`` (and ``transformer_exog`` when exog is present) so that
    the library handles standardisation itself. This avoids wrapping the
    estimator at the sklearn level — which previously collided with
    ``ForecasterRecursive._recursive_predict``'s LinearModel fast path and
    produced NaN cascades.
    """
    from sklearn.preprocessing import StandardScaler
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

    estimator = _build_estimator(model_name, model_params, random_seed=random_seed, logger=logger)

    needs_scaling = (model_name or "").strip().lower() in _NEEDS_SCALING
    transformer_y = StandardScaler() if needs_scaling else None
    transformer_exog = StandardScaler() if (needs_scaling and has_exog) else None

    return ForecasterRecursive(
        estimator=estimator,
        lags=n_lags,
        transformer_y=transformer_y,
        transformer_exog=transformer_exog,
    )


class SpotforecastTrainer:
    """Trains one ``ForecasterRecursive`` per target channel for a panel."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTrainer")

    def _load_panel_channel_config(self, panel_id: str) -> dict[str, Any]:
        """Load panel-specific channel model config YAML, if configured."""
        train_cfg = self.config.get("train", {})
        file_map = train_cfg.get("channel_config_files", {})
        if not isinstance(file_map, dict):
            return {}

        cfg_path_value = file_map.get(panel_id) or file_map.get(f"panel_{panel_id}")
        if not cfg_path_value:
            return {}

        cfg_path = Path(cfg_path_value)
        if not cfg_path.is_absolute():
            base_dir = Path(self.config.get("_config_base_dir", Path.cwd()))
            cfg_path = (base_dir / cfg_path).resolve()

        if not cfg_path.exists():
            raise FileNotFoundError(f"Channel model config for panel {panel_id} not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        if not isinstance(loaded, dict):
            raise ValueError(f"Channel model config for panel {panel_id} must be a mapping: {cfg_path}")
        return loaded

    def _get_weight_suffix(self) -> str:
        return _get_weight_suffix_from_config(self.config)

    def train_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        timestamp: str | None = None,
        save_model: bool = True,
        panel_specific_params: dict[str, Any] | None = None,
        channel_specific_params: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Train one ForecasterRecursive per target channel for a panel."""
        base_models_dir = Path(self.config["paths"]["models_dir"])
        train_ratio = self.config["train"]["train_ratio"]
        model_label = self.config["train"]["model"]
        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)
        # Predict Δy[t] = y[t] - y[t-1] instead of y[t]. At inference we
        # integrate one-step: y_pred[t] = y[t-1] + Δy_pred[t]. Makes the
        # target stationary so no model — degenerate or not — can win by
        # collapsing to the training mean. Set 0 to disable.
        diff_order = int(self.config.get("train", {}).get("differentiation", 1))

        if timestamp is None:
            timestamp = generate_timestamp()
        models_dir = base_models_dir / timestamp

        self.logger.info(f"Training spotforecast2 models on {len(df)} rows for panel {panel_id}")

        target_cols, exog_columns = _resolve_feature_columns(self.config, df)
        weight_residuals_enabled = self.config.get("exogenous", {}).get("weight_residuals", {}).get("enabled", False)
        if weight_residuals_enabled:
            self.logger.info("weight_residuals enabled: exogenous columns excluded from model features")

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            df = _mask_known_anomalies(df, known_anomalies, known_anomaly_buffer, columns=target_cols)

        train_df, test_df = _time_series_train_test_split(df, train_ratio=train_ratio)

        fallback_freq = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
        train_df = _ensure_datetime_freq(train_df, fallback_freq)
        test_df = _ensure_datetime_freq(test_df, fallback_freq)

        self.logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        panel_cfg_from_yaml = self._load_panel_channel_config(panel_id)
        if panel_specific_params:
            panel_cfg_from_yaml = {**panel_cfg_from_yaml, **panel_specific_params}

        panel_default_section = panel_cfg_from_yaml.get("default")
        if isinstance(panel_default_section, dict):
            panel_default_model = panel_default_section.get("model")
            panel_default_params = panel_default_section.get("params", {})
        else:
            panel_default_model = panel_cfg_from_yaml.get("default_model")
            panel_default_params = panel_cfg_from_yaml.get("default_params", {})
        channel_cfg_map = panel_cfg_from_yaml.get("channels", {})
        if channel_specific_params:
            channel_cfg_map = {**channel_cfg_map, **channel_specific_params}

        weight_suffix = self._get_weight_suffix()
        forecasters: dict[str, Any] = {}
        channel_model_specs: dict[str, dict[str, Any]] = {}
        y_pred_test = np.zeros((len(test_df), len(target_cols)))

        for i, target_col in enumerate(target_cols):
            channel_cfg = channel_cfg_map.get(target_col, {})
            if not isinstance(channel_cfg, dict):
                channel_cfg = {}

            channel_model_name = channel_cfg.get("model") or panel_default_model or model_label
            default_name_lc = (panel_default_model or model_label or "").strip().lower()
            channel_model_params: dict[str, Any] = {}
            if (channel_model_name or "").strip().lower() == default_name_lc:
                channel_model_params.update(panel_default_params or {})
            channel_model_params.update(channel_cfg.get("params", {}) or {})

            self.logger.info(f"  Training forecaster for: {target_col} (model={channel_model_name})")

            # Resolve per-channel lag spec from the panel YAML (best_lags), falling
            # back to the global train.lags. effective_lags is what we set on the
            # forecaster; effective_n_lags is the int used for sufficiency checks.
            effective_lags: Any = channel_cfg.get("best_lags", n_lags)
            if isinstance(effective_lags, (list, tuple)) and effective_lags:
                effective_n_lags = int(max(effective_lags))
                effective_lags = list(effective_lags)
            else:
                try:
                    effective_n_lags = int(effective_lags)
                    effective_lags = effective_n_lags
                except (TypeError, ValueError):
                    effective_n_lags = n_lags
                    effective_lags = n_lags

            y_train_raw = train_df[target_col].copy()
            y_train_raw.name = target_col

            observed_mask = _compute_observed_mask(train_df, target_col, weight_suffix)
            y_train = y_train_raw.copy()
            y_train.loc[~observed_mask] = np.nan
            y_train = _interpolate_for_model(y_train)
            if len(y_train) < effective_n_lags + 10:
                self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
                continue

            sample_mask_for_weights = None
            if self.config.get("train", {}).get("exclude_imputed_training_samples", False):
                sample_mask_for_weights = _build_strict_training_sample_mask(
                    observed_mask=observed_mask,
                    n_lags=effective_n_lags,
                )
                potential_rows = max(len(y_train) - effective_n_lags, 0)
                kept_rows = int(sample_mask_for_weights.iloc[effective_n_lags:].sum())
                self.logger.info(
                    f"    {target_col}: excluding {potential_rows - kept_rows} training sample(s) "
                    f"that contain imputed target/lag values"
                )
                if kept_rows == 0:
                    self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
                    continue

            train_cfg = self.config.get("train", {})
            if train_cfg.get("auto_clean_anomalies", False):
                anomaly_mask = _detect_anomalies_via_ridge(
                    y_train,
                    n_lags=effective_n_lags,
                    threshold_scale=train_cfg.get("auto_clean_threshold", 4.0),
                    buffer=train_cfg.get("auto_clean_buffer", 3),
                )
                n_flagged = int(anomaly_mask.sum())
                if n_flagged > 0:
                    self.logger.info(f"    {target_col}: auto-cleaning flagged {n_flagged} suspected anomaly points")
                    if sample_mask_for_weights is not None:
                        sample_mask_for_weights = sample_mask_for_weights & ~anomaly_mask
                    else:
                        sample_mask_for_weights = ~anomaly_mask

            exog_train = train_df[exog_columns].loc[y_train.index] if exog_columns else None

            forecaster = _create_forecaster(
                channel_model_name,
                channel_model_params,
                n_lags=effective_lags,
                random_seed=random_seed,
                has_exog=exog_train is not None,
                logger=self.logger,
            )

            if sample_mask_for_weights is not None:

                def _weight_func(index, mask=sample_mask_for_weights):
                    return mask.reindex(index).fillna(False).astype(float).to_numpy()

                forecaster.weight_func = _weight_func

            # Fit on Δy when differentiation > 0. spotforecast2's
            # transformer_y / weight_func still apply normally to the
            # differenced target.
            if diff_order > 0:
                y_for_fit = _difference(y_train, diff_order).dropna()
                exog_for_fit = exog_train.loc[y_for_fit.index] if exog_train is not None else None
            else:
                y_for_fit = y_train
                exog_for_fit = exog_train
            forecaster.fit(y=y_for_fit, exog=exog_for_fit)

            # Detach the closure so joblib can pickle the fitted forecaster.
            if sample_mask_for_weights is not None:
                forecaster.weight_func = None
                forecaster.source_code_weight_func = None

            channel_model_specs[target_col] = {
                "model": channel_model_name,
                "params": channel_model_params,
                "lags": effective_lags,
            }

            # One-step-ahead test predictions on real observed lags (mirrors
            # what detection / live mode does — see _predict_one_step_integrated).
            test_observed = _compute_observed_mask(test_df, target_col, weight_suffix)
            y_test_for_lags = test_df[target_col].copy()
            y_test_for_lags.loc[~test_observed] = np.nan
            y_test_for_lags = _interpolate_for_model(y_test_for_lags)

            full_y_raw = pd.concat([y_train, y_test_for_lags])
            full_y_raw.name = target_col

            full_exog = None
            if exog_columns:
                full_exog = pd.concat([train_df[exog_columns], test_df[exog_columns]])
                full_exog = _interpolate_for_model(full_exog)
                full_exog = full_exog.loc[full_y_raw.index]

            try:
                y_pred_test[:, i] = _predict_one_step_integrated(
                    forecaster, full_y_raw, full_exog, test_df.index, diff_order
                )
            except Exception as exc:
                self.logger.warning(f"  One-step-ahead test prediction failed for {target_col}: {exc}")
                y_pred_test[:, i] = np.nan

            forecasters[target_col] = forecaster

            y_test_col = test_df[target_col].values
            valid = ~np.isnan(y_test_col)
            if valid.any():
                rmse = np.sqrt(np.mean((y_test_col[valid] - y_pred_test[valid, i]) ** 2))
                mae = np.mean(np.abs(y_test_col[valid] - y_pred_test[valid, i]))
                self.logger.info(f"    {target_col}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        self.logger.info(f"Trained {len(forecasters)} forecasters")

        y_test_vals = test_df[target_cols].values
        rmse_results = np.sqrt(np.nanmean((y_test_vals - y_pred_test) ** 2, axis=0))
        mae_results = np.nanmean(np.abs(y_test_vals - y_pred_test), axis=0)

        res_df = pd.DataFrame({"rmse": rmse_results, "mae": mae_results}, index=target_cols)

        unique_channel_models = {spec["model"] for spec in channel_model_specs.values()}
        if len(unique_channel_models) == 1:
            actual_model_label = next(iter(unique_channel_models))
        elif len(unique_channel_models) > 1:
            actual_model_label = "Multi"
        else:
            actual_model_label = model_label

        model_data = {
            "forecasters": forecasters,
            "n_lags": n_lags,
            "target_cols": target_cols,
            "exog_columns": exog_columns,
            "channel_models": channel_model_specs,
            "differentiation": diff_order,
            "model_type": f"spotforecast2_{actual_model_label.lower()}",
            "model_name": actual_model_label,
            "train_ratio": train_ratio,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_start_timestamp": (
                train_df.index.min().isoformat()
                if len(train_df) > 0 and isinstance(train_df.index, pd.DatetimeIndex)
                else None
            ),
            "train_end_timestamp": (
                train_df.index.max().isoformat()
                if len(train_df) > 0 and isinstance(train_df.index, pd.DatetimeIndex)
                else None
            ),
            "test_start_timestamp": (
                test_df.index.min().isoformat()
                if len(test_df) > 0 and isinstance(test_df.index, pd.DatetimeIndex)
                else None
            ),
            "timestamp": timestamp,
        }

        if save_model:
            storage.ensure_dir(models_dir)
            model_path = models_dir / f"{model_label}_fc_model_panel_{panel_id}.pkl"
            joblib.dump(model_data, model_path)
            self.logger.info(f"Saved spotforecast2 model to {model_path}")

        self.logger.info(f"Average RMSE: {rmse_results.mean():.4f}")
        return res_df, timestamp

    def predict(
        self,
        model_data: dict[str, Any],
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Batch one-step-ahead predictions over ``df`` for anomaly scoring.

        Builds the lag/window-feature matrix from the *actually observed* values
        at every timestep (via ``forecaster.create_train_X_y``) and scores them
        in one pass, so each prediction is conditioned on real data — not on
        prior predictions, which is what the recursive ``predict`` would do.
        """
        forecasters = model_data["forecasters"]
        target_cols = model_data["target_cols"]
        exog_columns = model_data.get("exog_columns", [])
        # Older model artifacts (pre-differentiation) default to 0 = predict
        # raw y. Newer artifacts carry the actual order used at train time.
        diff_order = int(model_data.get("differentiation", 0))
        weight_suffix = self._get_weight_suffix()

        df = _ensure_datetime_freq(df)
        if history_df is not None:
            history_df = _ensure_datetime_freq(history_df)

        predictions: dict[str, np.ndarray] = {}

        for target_col in target_cols:
            forecaster = forecasters.get(target_col)
            if forecaster is None:
                predictions[target_col] = np.full(len(df), np.nan)
                continue

            if history_df is not None:
                full_y = pd.concat([history_df[target_col], df[target_col]])
                observed_mask = pd.concat(
                    [
                        _compute_observed_mask(history_df, target_col, weight_suffix),
                        _compute_observed_mask(df, target_col, weight_suffix),
                    ]
                )
            else:
                full_y = df[target_col].copy()
                observed_mask = _compute_observed_mask(df, target_col, weight_suffix)
            full_y.name = target_col

            full_y = full_y.copy()
            full_y.loc[~observed_mask] = np.nan
            full_y = _interpolate_for_model(full_y)
            if len(full_y) == 0:
                predictions[target_col] = np.full(len(df), np.nan)
                continue

            full_y = _ensure_datetime_freq(full_y)

            exog_full = None
            if exog_columns:
                cols_present = [c for c in exog_columns if c in df.columns]
                if cols_present:
                    if history_df is not None:
                        exog_full = pd.concat([history_df[cols_present], df[cols_present]])
                    else:
                        exog_full = df[cols_present].copy()
                    exog_full = exog_full.loc[full_y.index]
                    exog_full = _interpolate_for_model(exog_full)

            try:
                predictions[target_col] = _predict_one_step_integrated(
                    forecaster, full_y, exog_full, df.index, diff_order
                )
            except Exception as e:
                self.logger.warning(f"Prediction failed for {target_col}: {e}. Using NaN.")
                predictions[target_col] = np.full(len(df), np.nan)

        return pd.DataFrame(predictions, index=df.index)

    def train_all_panels(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        training_timestamp = generate_timestamp()
        return {pid: self.train_panel(pid, df, timestamp=training_timestamp) for pid, df in panel_data.items()}

    def run(self, panel_data: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
        self.logger.info("Starting spotforecast2 model training...")
        results = self.train_all_panels(panel_data)
        self.logger.info("spotforecast2 model training completed")
        return results


def _yaml_search_space_to_dict(raw: dict) -> dict:
    """Convert YAML search space to the dict format expected by spotoptim_search_forecaster.

    YAML lists of 2 elements become tuples (low, high), lists of 3 elements become
    (low, high, transform), and plain lists stay as categorical factors.
    """
    result = {}
    for key, value in raw.items():
        if isinstance(value, list):
            if len(value) in (2, 3) and all(isinstance(v, (int, float)) for v in value[:2]):
                result[key] = tuple(value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


# Replace NaN/Inf objective scores with this large but finite penalty so
# SpotOptim's GP surrogate never sees a non-finite y. Picked safely below
# float64's max so squared distances during GP kernel evaluation stay finite.
_NAN_PENALTY = 1e12


def _build_nan_safe_metric(metric_name: str):
    """Wrap an sklearn metric so a divergent trial returns a finite penalty.

    MLP / BayesianRidge / Huber occasionally return NaN preds under extreme
    hyperparameter samples. SpotOptim's default surrogate (GaussianProcessRegressor)
    refuses NaN/Inf in its training matrix, so a single bad trial otherwise aborts
    the entire (channel, model) search with ``ValueError: Input X contains NaN``.
    This wrapper turns those into ``_NAN_PENALTY`` — large enough that the
    optimiser steers away, finite enough that the GP fit succeeds.

    Higher-is-better metrics (``r2``) are negated so the minimiser still moves
    toward better fits.
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )

    lower_is_better = {
        "mean_absolute_error": mean_absolute_error,
        "mae": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "mse": mean_squared_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mape": mean_absolute_percentage_error,
    }
    higher_is_better = {
        "r2": r2_score,
        "r2_score": r2_score,
    }

    name_lc = metric_name.lower()
    if name_lc in lower_is_better:
        base = lower_is_better[name_lc]
        invert = False
    elif name_lc in higher_is_better:
        base = higher_is_better[name_lc]
        invert = True
    else:
        # Unknown metric — let spotforecast2 resolve the string itself.
        return metric_name

    def _safe_metric(y_true, y_pred, y_train=None, **kwargs):
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_pred_arr.size == 0 or y_true_arr.size == 0:
            return _NAN_PENALTY
        finite = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        if not finite.any():
            return _NAN_PENALTY
        try:
            value = float(base(y_true_arr[finite], y_pred_arr[finite]))
        except Exception:
            return _NAN_PENALTY
        if not np.isfinite(value):
            return _NAN_PENALTY
        return -value if invert else value

    _safe_metric.__name__ = metric_name
    return _safe_metric


def _build_raw_r2_under_differentiation(raw_y_full: pd.Series):
    """R² metric that reports the value in *raw y space* even when the
    forecaster is trained on Δy.

    When ``train.differentiation = 1`` the tuner's CV scores predictions
    on Δy by default, whose variance is much smaller than Var(y) for
    autocorrelated series — so ``R²_Δy`` is much smaller than the
    ``R²_raw`` that detection / live mode reports for the *same* model.
    To keep the tuner's leaderboard 1:1 with production, this metric:

    1. Recovers the raw truth on the val window by looking up the full raw
       series by timestamp (``y_true.index``).
    2. Recovers the residual scale: when the forecaster has a
       ``transformer_y`` (linear / kernel / MLP models), spotforecast2
       passes *scaled* Δy values to the metric. ``std(raw_Δy_train) /
       std(y_train_passed)`` reconstructs the StandardScaler's σ. For
       tree models there's no transformer, std(y_train_passed) =
       std(raw_Δy_train), so σ = 1 (no-op).
    3. Computes ``R² = 1 - SSE_raw / SST_raw`` with ``Var(y_raw)`` on the
       val window in the denominator — which is exactly what
       ``sklearn.r2_score`` would compute on integrated raw predictions.

    Same winner as Δy-R² (residual ranking is preserved), but the
    leaderboard number now matches what production reports for the
    same forecaster on the same window.
    """
    raw_dy_full = raw_y_full.diff()

    def _safe_metric(y_true, y_pred, y_train=None, **kwargs):
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        val_size = y_pred_arr.size
        if val_size == 0 or y_true_arr.size == 0:
            return _NAN_PENALTY
        # spotoptim_search_forecaster strips the index and passes raw
        # ndarrays; backtesting_forecaster keeps Series. Use positional
        # (iloc) lookup so both paths work — the val window is always the
        # *last* ``val_size`` rows of the differenced series (that's how
        # OneStepAheadFold slices), so the raw truth window is the same
        # tail of raw_y_full.
        if val_size > len(raw_y_full):
            return _NAN_PENALTY
        raw_truth = raw_y_full.iloc[-val_size:].to_numpy()

        # Recover σ exactly via the linear relationship StandardScaler
        # enforces: ``raw_dy = σ × scaled + μ``. The training portion
        # passed to the metric corresponds to the rows of raw_dy_full
        # immediately *before* the val window (spotforecast2 trims the
        # left of training by window_size twice, but those trimmed rows
        # are at the start, not the end — so the last ``train_size`` rows
        # before val align exactly with what was passed). For tree models
        # (no transformer_y) the relationship is the identity → σ = 1.
        sigma = 1.0
        if y_train is not None:
            y_train_arr = np.asarray(y_train, dtype=float)
            train_size = y_train_arr.size
            val_start_iloc = len(raw_dy_full) - val_size
            train_lo = val_start_iloc - train_size
            if train_size > 1 and train_lo >= 0:
                raw_dy_at_train = raw_dy_full.iloc[train_lo:val_start_iloc].to_numpy()
                ok = np.isfinite(y_train_arr) & np.isfinite(raw_dy_at_train)
                if ok.sum() > 1:
                    a = np.column_stack([y_train_arr[ok], np.ones(int(ok.sum()))])
                    coefs, *_ = np.linalg.lstsq(a, raw_dy_at_train[ok], rcond=None)
                    derived = float(coefs[0])
                    if np.isfinite(derived) and derived != 0.0:
                        sigma = derived

        resid_raw = (y_true_arr - y_pred_arr) * sigma
        finite = np.isfinite(resid_raw) & np.isfinite(raw_truth)
        if not finite.any():
            return _NAN_PENALTY
        truth_f = raw_truth[finite]
        resid_f = resid_raw[finite]
        sst = float(np.sum((truth_f - truth_f.mean()) ** 2))
        if sst <= 0.0:
            return _NAN_PENALTY
        sse = float(np.sum(resid_f**2))
        r2 = 1.0 - sse / sst
        if not np.isfinite(r2):
            return _NAN_PENALTY
        return -r2  # higher R² is better; minimiser needs the negation

    _safe_metric.__name__ = "r2_raw_after_integration"
    return _safe_metric


class SpotforecastTuner:
    """Tunes ForecasterRecursive hyperparameters per (channel, model) via SpotOptim.

    Uses ``spotoptim_search_forecaster`` from ``spotforecast2.model_selection``
    to run surrogate-model Bayesian optimisation per target channel for every
    candidate model in ``tune_config['models']``, picking the per-channel winner.
    """

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("SpotforecastTuner")

    def _get_weight_suffix(self) -> str:
        return _get_weight_suffix_from_config(self.config)

    def tune_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        tune_config: dict[str, Any],
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Tune hyperparameters for each channel in a panel via SpotOptim.

        Returns
        -------
        Dict mapping channel_name -> ``{best_params, best_lags, best_metric,
        best_model, model_scores}``. ``model_scores`` is a sub-dict per
        candidate model so callers can compare them, not just the winner.
        """
        from spotforecast2.model_selection import OneStepAheadFold, spotoptim_search_forecaster
        from tqdm.auto import tqdm

        n_lags = self.config["train"].get("lags", 24)
        random_seed = self.config["train"].get("random_seed", 42)
        exclude_imputed_training_samples = self.config.get("train", {}).get("exclude_imputed_training_samples", False)
        # Tune on Δy too — must match what train_panel fits, otherwise the
        # winning hyperparameters won't be optimal for the actual estimator
        # that ends up in production.
        diff_order = int(self.config.get("train", {}).get("differentiation", 1))
        # Conservative cap for the strict-sample mask: must cover the longest
        # lag the search may pick, so use the global config value as an upper
        # bound (search-space lag lists override per trial).
        if isinstance(n_lags, (list, tuple)) and n_lags:
            n_lags_max = int(max(n_lags))
        else:
            try:
                n_lags_max = int(n_lags)
            except (TypeError, ValueError):
                n_lags_max = 24

        known_anomalies = self.config.get("known_anomalies", [])
        known_anomaly_buffer = self.config["train"].get("known_anomaly_buffer")
        if known_anomalies and known_anomaly_buffer:
            df = _mask_known_anomalies(df, known_anomalies, known_anomaly_buffer)

        # Match SpotforecastTrainer.train_panel exactly: include `exogenous_*`
        # columns as exog features and exclude them from the tuning targets.
        # Otherwise tune optimizes a different model than train ends up fitting.
        target_cols, exog_columns = _resolve_feature_columns(self.config, df)
        weight_suffix = self._get_weight_suffix()

        if channels is not None:
            target_cols = [c for c in target_cols if c in channels]
            missing = set(channels) - set(target_cols)
            if missing:
                self.logger.warning(f"Requested channels not found in data: {missing}")

        # Use ALL the processed data — the trailing 10–20% carries the
        # strongest distribution drift; chopping it off would make CV blind
        # to live-shift. OneStepAheadFold below provides the train/val split.
        tune_df = _ensure_datetime_freq(df, self.config.get("process", {}).get("resample", {}).get("freq", "5min"))

        n_trials = tune_config.get("n_trials", 10)
        n_initial = tune_config.get("n_initial", 5)
        metric = tune_config.get("metric", "mean_absolute_error")
        # Wrap the string metric so a single divergent MLP / Bayesian / Huber
        # trial doesn't crash SpotOptim's surrogate fit (GaussianProcessRegressor
        # rejects NaN/Inf). Falls through unchanged when metric is already a
        # callable or an unknown string.
        metric_callable = _build_nan_safe_metric(metric) if isinstance(metric, str) else metric
        model_search_spaces = tune_config.get("model_search_spaces", {})

        models = tune_config.get("models", ["LightGBM"])
        if isinstance(models, str):
            models = [models]
        # Validate names via _build_estimator. Log the exact cause so a typo
        # or a missing dependency doesn't silently collapse the run to trees.
        valid_models: list[str] = []
        for m in models:
            try:
                _build_estimator(m, {}, random_seed=0, logger=self.logger)
                valid_models.append(m)
            except ValueError as exc:
                self.logger.warning(f"Skipping model '{m}': {exc}")
            except ImportError as exc:
                self.logger.warning(f"Skipping model '{m}': dependency missing ({exc})")
        models = valid_models or ["LightGBM"]

        self.logger.info(
            f"Tuning panel {panel_id}: {len(target_cols)} channels, "
            f"{n_trials} trials, {n_initial} initial, models={models}"
        )

        # OneStepAheadFold matches what production actually does at detection
        # time (see SpotforecastTrainer.predict). Train on the first 80%,
        # validate on the most recent 20% — that val window now spans the
        # part of the timeline closest to live conditions, so any
        # distribution drift inside the dataset shows up in the metric and
        # constant-mean predictors get exposed instead of tying real models.
        cv = OneStepAheadFold(initial_train_size=max(1, int(len(tune_df) * 0.8)), verbose=False)

        results: dict[str, dict[str, Any]] = {}

        channel_pbar = tqdm(target_cols, desc=f"Panel {panel_id}", unit="channel", leave=True)
        for target_col in channel_pbar:
            channel_pbar.set_postfix_str(target_col)

            channel_n_trials = n_trials
            channel_n_initial = n_initial
            panel_overrides = tune_config.get("panel_overrides", {}).get(panel_id, {})
            if panel_overrides.get("default"):
                channel_n_trials = panel_overrides["default"].get("n_trials", channel_n_trials)
                channel_n_initial = panel_overrides["default"].get("n_initial", channel_n_initial)
            channel_overrides = panel_overrides.get("channels", {}).get(target_col, {})
            if channel_overrides:
                channel_n_trials = channel_overrides.get("n_trials", channel_n_trials)
                channel_n_initial = channel_overrides.get("n_initial", channel_n_initial)

            # Match train: respect imputation `__weight` flags so imputed rows
            # don't masquerade as observations during the search.
            observed_mask = _compute_observed_mask(tune_df, target_col, weight_suffix)
            y_train_raw = tune_df[target_col].copy()
            y_train_raw.name = target_col
            y_train = y_train_raw.copy()
            y_train.loc[~observed_mask] = np.nan
            y_train = _interpolate_for_model(y_train)
            if len(y_train) < n_lags_max + 10:
                self.logger.warning(f"  Skipping {target_col}: insufficient data ({len(y_train)} rows)")
                continue

            sample_mask_for_weights: pd.Series | None = None
            if exclude_imputed_training_samples:
                sample_mask_for_weights = _build_strict_training_sample_mask(
                    observed_mask=observed_mask,
                    n_lags=n_lags_max,
                )
                if int(sample_mask_for_weights.iloc[n_lags_max:].sum()) == 0:
                    self.logger.warning(f"  Skipping {target_col}: no fully observed training samples left")
                    continue

            train_cfg_section = self.config.get("train", {})
            if train_cfg_section.get("auto_clean_anomalies", False):
                anomaly_mask = _detect_anomalies_via_ridge(
                    y_train,
                    n_lags=n_lags_max,
                    threshold_scale=train_cfg_section.get("auto_clean_threshold", 4.0),
                    buffer=train_cfg_section.get("auto_clean_buffer", 3),
                )
                if int(anomaly_mask.sum()) > 0:
                    if sample_mask_for_weights is not None:
                        sample_mask_for_weights = sample_mask_for_weights & ~anomaly_mask
                    else:
                        sample_mask_for_weights = ~anomaly_mask

            exog_train = None
            if exog_columns:
                exog_train = tune_df[exog_columns].loc[y_train.index]
                exog_train = _interpolate_for_model(exog_train)

            # Predict Δy[t] just like train_panel will. The tuner must score
            # candidates on the same target the production model will fit
            # against, otherwise the winning hyperparameters are optimal
            # for a different objective.
            channel_metric_callable = metric_callable
            if diff_order > 0:
                # Capture the raw (pre-difference) series — the R² metric
                # looks up actual raw values by timestamp inside the val
                # window so the leaderboard number matches production.
                y_train_raw_pre_diff = y_train.copy()
                y_train = _difference(y_train, diff_order).dropna()
                if exog_train is not None:
                    exog_train = exog_train.loc[y_train.index]
                if sample_mask_for_weights is not None:
                    sample_mask_for_weights = sample_mask_for_weights.reindex(y_train.index).fillna(False)
                # Only R² needs the raw-space rescaling — MAE/MSE/RMSE
                # numerators are the same residuals in either space, so
                # they already match production without modification.
                if isinstance(metric, str) and metric.lower() in ("r2", "r2_score") and diff_order == 1:
                    channel_metric_callable = _build_raw_r2_under_differentiation(y_train_raw_pre_diff)

            best_overall: dict[str, Any] | None = None
            per_model: dict[str, dict[str, Any]] = {}

            model_pbar = tqdm(models, desc=f"  {target_col}", unit="model", leave=False)
            for model_name in model_pbar:
                model_pbar.set_postfix_str(model_name)

                try:
                    forecaster = _create_forecaster(
                        model_name,
                        {},
                        n_lags=n_lags,
                        random_seed=random_seed,
                        has_exog=exog_train is not None,
                        logger=self.logger,
                    )
                    search_space = _yaml_search_space_to_dict(model_search_spaces.get(model_name, {}))

                    # Match train: zero out imputed rows during fit.
                    if sample_mask_for_weights is not None:

                        def _weight_func(index, mask=sample_mask_for_weights):
                            return mask.reindex(index).fillna(False).astype(float).to_numpy()

                        forecaster.weight_func = _weight_func

                    tuning_results, _ = spotoptim_search_forecaster(
                        forecaster=forecaster,
                        y=y_train,
                        cv=cv,
                        search_space=search_space,
                        metric=channel_metric_callable,
                        exog=exog_train,
                        return_best=True,
                        random_state=random_seed,
                        verbose=False,
                        n_trials=channel_n_trials,
                        n_initial=channel_n_initial,
                        show_progress=False,
                    )

                    best_row = tuning_results.iloc[0]
                    best_params = best_row["params"] if "params" in tuning_results.columns else {}
                    best_lags = best_row["lags"] if "lags" in tuning_results.columns else n_lags

                    non_meta_cols = [
                        c for c in tuning_results.columns if c not in ("params", "lags") and c not in best_params
                    ]
                    best_metric_val = None
                    if non_meta_cols:
                        raw_val = best_row[non_meta_cols[0]]
                        if hasattr(raw_val, "__len__") and not isinstance(raw_val, str):
                            best_metric_val = float(raw_val[0])
                        else:
                            best_metric_val = float(raw_val)

                    self.logger.info(
                        f"    {target_col} [{model_name}]: {metric}={best_metric_val:.6f}, lags={best_lags}"
                    )

                    per_model[model_name] = {
                        "best_metric": best_metric_val,
                        "best_params": best_params,
                        "best_lags": best_lags,
                    }

                    if best_overall is None or (
                        best_metric_val is not None and best_metric_val < best_overall["best_metric"]
                    ):
                        best_overall = {
                            "best_params": best_params,
                            "best_lags": best_lags,
                            "best_metric": best_metric_val,
                            "best_model": model_name,
                        }

                except Exception as e:
                    self.logger.error(
                        f"  Tuning failed for {target_col} [{model_name}]: {e}",
                        exc_info=True,
                    )
                    per_model[model_name] = {"error": str(e)}

            model_pbar.close()

            if best_overall is not None:
                results[target_col] = {**best_overall, "model_scores": per_model}
                self.logger.info(
                    f"  >> {target_col}: winner={best_overall['best_model']}, "
                    f"{metric}={best_overall['best_metric']:.6f}"
                )
            else:
                results[target_col] = {"error": "All models failed", "model_scores": per_model}

        channel_pbar.close()
        return results
