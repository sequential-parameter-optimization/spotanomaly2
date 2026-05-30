"""Pre-fit data shaping helpers shared by trainer and tuner.

Pure functions over panel DataFrames / target Series — no model state, no I/O.
Handles the bits spotanomaly2 needs that ``MultiTask.prepare_data`` doesn't
cover: known-anomaly masking with a buffer, target/exog column split,
imputation-flag-aware observation masks, a strict mask that requires all lag
inputs to be observed too, a Ridge-residual anomaly pre-pass for training-region
cleanup, and the small frequency/interpolation conveniences used by the
predict-path helpers.
"""

from typing import Any

import numpy as np
import pandas as pd

from spotanomaly2.domain import imputation_methods
from spotanomaly2.infrastructure import logging


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
    existing_cols = [c for c in cols_to_mask if c in df.columns]
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
            df.loc[mask, existing_cols] = np.nan
        except (ValueError, TypeError) as exc:
            logging.get_logger().error("Failed to mask known anomaly window: %s", exc)
            continue
    return df


def _apply_known_anomaly_imputation(
    df: pd.DataFrame,
    known_anomalies: list[dict],
    buffer: str,
    target_cols: list[str],
    weight_suffix: str,
    imputation_method: str = "linear_interpolation",
    imputation_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Blank known-anomaly windows on ``target_cols``, re-impute, merge ``__weight``.

    The trainer/tuner need a gap-free target series so autoregressive lag
    features are computable, but they also need a way to keep known anomalies
    out of the loss. This helper does both in one pass:

    1. Blank target cells inside known-anomaly windows (extended by ``buffer``).
    2. Re-impute with ``imputation_method`` — same method the process stage
       uses, so the imputation strategy is single-sourced from config.
    3. Set the matching ``{target}{weight_suffix}`` column to 0 for the
       blanked rows, AND-ed with any pre-existing weight from the process
       stage.

    After this, ``__weight`` uniformly marks "this row is not real" for any
    reason (process-imputed or known-anomaly), so ``_compute_observed_mask`` and
    ``_build_strict_training_sample_mask`` do the right thing without any
    further branching downstream.

    The legacy ``psm`` method isn't registered in
    ``imputation_methods.IMPUTATION_METHODS``; for the small interior cells from
    known-anomaly masking, fall back to ``linear_interpolation``.
    """
    if not known_anomalies or not buffer or not target_cols:
        return df

    imputation_params = dict(imputation_params or {})
    if imputation_method not in imputation_methods.IMPUTATION_METHODS:
        imputation_method = "linear_interpolation"
        imputation_params = {}

    masked = _mask_known_anomalies(df, known_anomalies, buffer, columns=target_cols)

    for col in target_cols:
        # Cells the mask blanked: NaN now AND not NaN in the input. This is the
        # purely additive contribution of known-anomaly masking on top of
        # whatever the process stage already left behind.
        anomaly_nan = masked[col].isna() & ~df[col].isna()
        if not anomaly_nan.any():
            continue

        weight_col = f"{col}{weight_suffix}"
        pre_weight = (
            masked[weight_col].fillna(0).astype(int)
            if weight_col in masked.columns
            else pd.Series(1, index=masked.index, dtype=int)
        )
        masked[col] = imputation_methods.impute_series(masked[col], method=imputation_method, **imputation_params)
        masked[weight_col] = (pre_weight & (~anomaly_nan).astype(int)).astype(int)

    return masked


def _split_panel_columns(
    df: pd.DataFrame,
    configured_exog_columns: list[str],
    weight_suffix: str,
    multiplier_prefixes: list[str],
) -> tuple[list[str], list[str]]:
    """Split DataFrame columns into (target_cols, exog_columns).

    Mirrors the logic ``SpotforecastTrainer.train_panel`` uses so the tuner sees
    the same model topology as the training step: ``exogenous_*`` columns are fed
    in as exog features and are NOT scored as targets. Columns of a source with
    ``multiply_residuals`` (identified by ``multiplier_prefixes``) are neither —
    they multiply detection residuals, so they are excluded from features too.
    """
    prefixed_exog_columns = [
        col
        for col in df.columns
        if col.startswith("exogenous_")
        and not col.endswith(weight_suffix)
        and not any(col.startswith(p) for p in multiplier_prefixes)
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
    """Return True only when target and all required lag inputs are observed.

    Equivalent to ANDing ``observed_mask`` with each of its 1..n_lags shifts,
    expressed as a single backward-looking rolling-min over a window of size
    ``n_lags + 1``. For 0/1 inputs, ``min == 1`` iff every value in the window
    is True. Incomplete leading windows (positions ``< n_lags``) become False.
    """
    obs = observed_mask.fillna(False).astype(int)
    rolled = obs.rolling(n_lags + 1, min_periods=n_lags + 1).min()
    return rolled.fillna(0).astype(bool)


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


def _ensure_freq(obj, fallback_freq: str | None = None):
    """Set ``obj.index.freq`` from ``pd.infer_freq``, falling back to
    ``fallback_freq`` when inference fails. Works on Series and DataFrame.
    """
    if isinstance(obj.index, pd.DatetimeIndex) and obj.index.freq is None:
        freq = pd.infer_freq(obj.index) or fallback_freq
        if freq:
            obj = obj.asfreq(freq)
    return obj


def _interpolate_inplace(obj):
    """Time-aware interpolation + bfill/ffill for Series or DataFrame."""
    if isinstance(obj.index, pd.DatetimeIndex):
        obj = obj.interpolate(method="time").bfill().ffill()
    else:
        obj = obj.interpolate(method="linear").bfill().ffill()
    return obj
