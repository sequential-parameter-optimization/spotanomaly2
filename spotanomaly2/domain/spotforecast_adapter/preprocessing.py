"""Pre-fit data shaping helpers shared by trainer and tuner.

Pure functions over panel DataFrames / target Series — no model state, no I/O.
Handles the bits spotanomaly2 needs that ``MultiTask.prepare_data`` doesn't
cover: known-anomaly masking with a buffer, imputation-flag-aware observation
masks, a strict mask that requires all lag inputs to be observed too, a
Ridge-residual anomaly pre-pass for training-region cleanup, and the small
frequency/interpolation conveniences used by the predict-path helpers.

Target/exog column-role resolution lives in ``panel_layout``.
"""

from typing import Any

import numpy as np
import pandas as pd


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
