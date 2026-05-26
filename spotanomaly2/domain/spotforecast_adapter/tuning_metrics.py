"""Tuner-specific metric wrappers and search-space conversion.

- ``_yaml_search_space_to_dict``: convert a YAML-friendly representation of a
  spotoptim search space (lists of length 2/3 for ranges, plain lists for
  categoricals) into the tuple form spotoptim expects.
- ``_NAN_PENALTY``: large-but-finite value SpotOptim's GP surrogate can accept
  in place of NaN/Inf scores produced by divergent trials.
- ``_build_nan_safe_metric``: wrap a string metric so a single divergent trial
  doesn't abort the whole search.
- ``_build_raw_r2_under_differentiation``: R² reported in raw-y space even when
  the forecaster fits Δy, so the tuner leaderboard matches production scoring.
"""

import numpy as np
import pandas as pd

# Replace NaN/Inf objective scores with this large but finite penalty so
# SpotOptim's GP surrogate never sees a non-finite y. Picked safely below
# float64's max so squared distances during GP kernel evaluation stay finite.
_NAN_PENALTY = 1e12


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
