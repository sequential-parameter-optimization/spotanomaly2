# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bayesian hyperparameter tuning for anomaly scorers via SpotOptim.

Uses ``spotoptim.SpotOptim`` (Kriging surrogate + sequential acquisition)
directly with a closure that maps a candidate parameter dict to negated
PR-AUC on the labelled benchmark dataset. This is genuinely
surrogate-based search — not grid search relabeled.

The two helpers ``convert_search_space`` / ``array_to_params`` are
duplicated from ``spotforecast2.model_selection.spotoptim_search`` (see
``_spotoptim_helpers.py``) so we don't pull in forecaster-specific
plumbing that lives next to them upstream.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from spotoptim import SpotOptim

from spotanomaly2.research._spotoptim_helpers import array_to_params, convert_search_space
from spotanomaly2.research.scorer_runner import run_scorer

LOGGER = logging.getLogger(__name__)

# A penalty value returned to the optimizer when a trial errors. Must be
# strictly larger than any plausible "good" value (negated PR-AUC ∈ [-1, 0]),
# so the surrogate steers away from the failing region.
_TRIAL_FAILURE_PENALTY = 1.0


def tune_scorer(
    scorer_name: str,
    search_space: dict[str, Any],
    y_true_train: pd.DataFrame,
    y_pred_train: pd.DataFrame,
    y_true_test: pd.DataFrame,
    y_pred_test: pd.DataFrame,
    y_true_labels: np.ndarray,
    *,
    n_trials: int = 30,
    n_initial: int = 8,
    high_quantile: float = 0.995,
    fixed_params: dict[str, Any] | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune one scorer's hyperparameters by maximising PR-AUC.

    Args:
        scorer_name: One of the four supported scorer names (see
            :func:`run_scorer`).
        search_space: Dict mapping parameter name → spec. Specs follow
            the same convention as
            ``spotforecast2.model_selection.spotoptim_search``:
            ``(low, high)`` float/int, ``(low, high, "log10")`` for
            log-scale floats, or a ``list`` for categorical factors.
        y_true_*, y_pred_*, y_true_labels: Inputs forwarded to
            :func:`run_scorer` for every trial.
        n_trials: Total budget (initial design + sequential trials).
        n_initial: Size of the initial space-filling design.
        high_quantile: Pinned operating threshold passed to every trial.
        fixed_params: Parameters held constant across all trials (merged
            into each trial's params dict).
        random_state: Seed for SpotOptim's design + surrogate.
        verbose: SpotOptim verbosity flag.

    Returns:
        Tuple ``(best_params, trials_df)``:
            * ``best_params`` — typed dict of the highest-PR-AUC point
              evaluated, with ``fixed_params`` merged in.
            * ``trials_df`` — one row per evaluated trial: ``trial_idx``,
              the parameter values, ``pr_auc``, and ``objective``
              (= -PR-AUC).
    """
    fixed_params = dict(fixed_params or {})

    bounds, var_type, var_name, var_trans = convert_search_space(search_space)

    trials: list[dict[str, Any]] = []

    def objective(X: np.ndarray) -> np.ndarray:
        # SpotOptim hands us X of shape (batch, n_params).
        out = np.empty(X.shape[0], dtype=float)
        for i, row in enumerate(X):
            params = array_to_params(row, var_name, var_type, bounds)
            params = {**fixed_params, **params}
            try:
                metrics = run_scorer(
                    scorer_name,
                    params,
                    y_true_train=y_true_train,
                    y_pred_train=y_pred_train,
                    y_true_test=y_true_test,
                    y_pred_test=y_pred_test,
                    y_true_labels=y_true_labels,
                    high_quantile=high_quantile,
                    n_resamples_bootstrap=0,  # skip CIs during tuning
                )
                pr = metrics.get("pr_auc")
                if pr is None or np.isnan(pr):
                    out[i] = _TRIAL_FAILURE_PENALTY
                else:
                    out[i] = -float(pr)
                trials.append(
                    {
                        "trial_idx": len(trials),
                        **params,
                        "pr_auc": float(pr) if pr is not None and not np.isnan(pr) else float("nan"),
                        "objective": float(out[i]),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — tuner should never crash on a bad point
                LOGGER.warning("Trial failed for %s with %r: %s", scorer_name, params, exc)
                out[i] = _TRIAL_FAILURE_PENALTY
                trials.append(
                    {
                        "trial_idx": len(trials),
                        **params,
                        "pr_auc": float("nan"),
                        "objective": _TRIAL_FAILURE_PENALTY,
                        "error": str(exc),
                    }
                )
        return out

    optimizer = SpotOptim(
        fun=objective,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        var_trans=var_trans,
        max_iter=n_trials,
        n_initial=n_initial,
        seed=random_state,
        verbose=verbose,
    )
    result = optimizer.optimize()

    best_params = array_to_params(result.x, var_name, var_type, bounds)
    best_params = {**fixed_params, **best_params}

    trials_df = pd.DataFrame(trials)
    return best_params, trials_df
