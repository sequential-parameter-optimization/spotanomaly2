# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for the per-channel scorer benchmark notebook.

This module is intentionally minimal — it exposes a handful of utility
functions the notebook needs and nothing else. Nothing here is wired into
the production detection path; the notebook is research-only and the
production scorer choice stays in ``config/default.yaml::detect``.

Public surface:

- :func:`mask_known_anomalies` — thin wrapper around the existing helper
  in ``spotanomaly2.domain.spotforecast_adapter`` so the notebook can mask
  windows without reaching into a private name.
- :func:`robust_scale` — MAD- and IQR-based robust scale estimate; used in
  the per-channel diagnostic cell to flag residual scales that look off.
- :func:`recall_at_precision` — operational point on the PR curve.
- :func:`cohens_kappa` — agreement between two binary flag vectors.
- :func:`bootstrap_pr_auc_ci` — bootstrap CI for PR-AUC.
- :func:`compute_residuals` — train a forecaster on the first 80% of a
  panel, predict on the last 20%, and return the residual matrix.
"""

from __future__ import annotations

import logging as _stdlogging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_curve

from spotanomaly2.domain.spotforecast_adapter import (
    SpotforecastPredictor,
    SpotforecastTrainer,
    _mask_known_anomalies,
)


def mask_known_anomalies(
    df: pd.DataFrame,
    known_anomalies: list[dict],
    buffer: str = "1d",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """NaN-out rows that fall inside known-anomaly windows (extended by ``buffer``).

    Re-exports the existing helper in ``spotanomaly2.domain.spotforecast_adapter``
    so the notebook does not import from a private name.
    """
    return _mask_known_anomalies(df, known_anomalies, buffer=buffer, columns=columns)


def robust_scale(values: np.ndarray | pd.Series) -> float:
    """Return a robust scale estimate for a 1-D vector.

    Computes both MAD × 1.4826 and IQR / 1.349 (the two are equivalent for
    Gaussian samples) and returns the larger of the two. Both estimators
    are insensitive to a small fraction of contaminating outliers — which
    is what we want here, because the underlying real-data residuals are
    known to contain latent anomalies that would inflate ``std``.

    Returns ``nan`` for empty input or for vectors where every value is
    NaN; returns ``0.0`` when the input is constant (no spread).
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    return max(mad * 1.4826, iqr / 1.349)


def recall_at_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_precision: float = 0.9,
) -> float:
    """Maximum recall achievable at ``precision >= target_precision``.

    Reads off the PR curve: for each threshold sklearn returns a
    (precision, recall) pair; we take the highest recall whose precision
    meets the target. Returns ``nan`` if the target precision is never
    reached (the scorer cannot produce a high-precision operating point)
    or if ``y_true`` is single-class.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, s)
    eligible = precision >= target_precision
    if not eligible.any():
        return float("nan")
    return float(recall[eligible].max())


def cohens_kappa(flags_a: np.ndarray, flags_b: np.ndarray) -> float:
    """Cohen's κ between two binary flag vectors.

    Returns ``nan`` if either vector is empty or single-valued (κ is
    undefined when one rater never flags anything).
    """
    a = np.asarray(flags_a, dtype=int).ravel()
    b = np.asarray(flags_b, dtype=int).ravel()
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = 200,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap PR-AUC with a (low, high) confidence interval.

    Returns ``(point_estimate, ci_low, ci_high)``. Point estimate is the
    PR-AUC on the original sample; bootstrap percentiles come from
    ``n_resamples`` resamples with replacement. Returns ``(nan, nan, nan)``
    if ``y_true`` is single-class.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if len(np.unique(y)) < 2 or len(y) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(42)
    point = float(average_precision_score(y, s))
    samples = np.empty(n_resamples, dtype=float)
    n = len(y)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        s_b = s[idx]
        if len(np.unique(y_b)) < 2:
            samples[i] = np.nan
            continue
        samples[i] = float(average_precision_score(y_b, s_b))
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return (point, float("nan"), float("nan"))
    alpha = (1.0 - ci) / 2.0
    return (
        point,
        float(np.percentile(samples, 100 * alpha)),
        float(np.percentile(samples, 100 * (1.0 - alpha))),
    )


def compute_residuals(
    panel_id: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: _stdlogging.Logger | None = None,
    train_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit forecaster on first ``train_ratio`` of ``df`` and return residuals.

    Trains a fresh ``SpotforecastTrainer`` model in a temporary models
    directory (so ``data/models/`` is left untouched), then uses the same
    trainer to one-step-ahead-predict the held-out test split. Returns
    ``(y_true_test, y_pred_test, residuals)`` — all DataFrames aligned on
    the test split's index, with one column per target channel.

    Notebooks should cache these to parquet rather than re-calling this
    function across cells; a full panel refit takes minutes.
    """
    cfg = dict(config)
    cfg["paths"] = dict(config.get("paths", {}))
    cfg["train"] = dict(config.get("train", {}))
    if train_ratio is not None:
        cfg["train"]["train_ratio"] = float(train_ratio)
    train_ratio_eff = float(cfg["train"].get("train_ratio", 0.8))

    with tempfile.TemporaryDirectory(prefix="per_channel_models_") as tmpdir:
        cfg["paths"]["models_dir"] = tmpdir
        trainer = SpotforecastTrainer(cfg, logger=logger)
        _, timestamp = trainer.train_panel(panel_id, df, save_model=True)

        model_path = Path(tmpdir) / timestamp / f"fc_model_panel_{panel_id}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at expected path: {model_path}")
        model_data = joblib.load(model_path)

        split_idx = int(len(df) * train_ratio_eff)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        predictor = SpotforecastPredictor(cfg, logger=logger)
        y_pred_test = predictor.predict(model_data, test_df, history_df=train_df)
        # Align y_true to the columns the forecaster actually produced
        # (exog columns are dropped from target_cols inside the trainer).
        target_cols = list(y_pred_test.columns)
        y_true_test = test_df[target_cols].copy()
        residuals = y_true_test.subtract(y_pred_test)

    return y_true_test, y_pred_test, residuals
