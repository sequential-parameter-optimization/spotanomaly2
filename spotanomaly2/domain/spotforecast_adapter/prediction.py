"""One-step-ahead prediction path + differentiation helpers.

These are shared between train-time evaluation and ``SpotforecastTrainer.predict``
so the metrics reported during training match what detection / live mode see.

- ``_difference`` / ``_integrate_one_step``: applied when ``train.differentiation``
  is on, so the forecaster fits Δy and we integrate one step against the real
  y[t-1] at inference.
- ``_one_step_ahead_predict``: bypasses ``ForecasterRecursive``'s recursive
  predict (which cascade-diverges over long horizons for non-tree models) and
  hits the bare estimator with lag features built from observed values.
- ``_predict_one_step_integrated``: combines the two; the entry point used by
  trainer.predict and train-time test eval.
"""

import numpy as np
import pandas as pd

from .preprocessing import _ensure_freq


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

    full_y = _ensure_freq(full_y)
    if full_exog is not None:
        full_exog = _ensure_freq(full_exog)

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
