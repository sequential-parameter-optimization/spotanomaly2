# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluation metrics for the scorer benchmark.

Threshold-independent metrics (PR-AUC, ROC-AUC, AP) form the cross-scorer
leaderboard; threshold-dependent metrics (precision, recall, F1, MCC) are
reported per-scorer at that scorer's tuned operating point only. See the
plan's risk-mitigation note on cross-scorer scale differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe(metric_fn, *args, default: float = float("nan"), **kwargs) -> float:
    """Call a sklearn metric; return ``default`` if it raises (degenerate input)."""
    try:
        return float(metric_fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision — the staircase area under the precision-recall curve.

    AP is what is conventionally reported as "PR-AUC" in the anomaly
    detection literature; it equals 1.0 for perfect separation and the
    positive-class prevalence for random scores. Used as the primary
    cross-scorer leaderboard metric and as the (negated) tuning objective.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def bootstrap_f1_ci(
    y_true: np.ndarray,
    flags: np.ndarray,
    n_resamples: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """95% bootstrap CI for F1 — caller still needs to compute the point estimate."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = _safe(f1_score, y_true[idx], flags[idx], zero_division=0)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    flags: np.ndarray,
    *,
    fit_seconds: float | None = None,
    score_seconds: float | None = None,
    n_resamples_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compute the full metric bundle for a single scorer run.

    Args:
        y_true: Binary ground-truth labels (0/1), 1-D.
        scores: Raw anomaly scores (higher = more anomalous), 1-D.
        flags: Hard binary flags from the detector (0/1), 1-D.
        fit_seconds: Wall-clock fit time, optional.
        score_seconds: Wall-clock score time, optional.
        n_resamples_bootstrap: Bootstrap iterations for F1 CI.
        rng: Optional generator for reproducible bootstrap.

    Returns:
        Dictionary keyed by metric name. NaN values mean the metric is
        undefined for the inputs (e.g. all-zero ``y_true``).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    flags = np.asarray(flags).astype(int).ravel()

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    prevalence = float(n_pos / max(len(y_true), 1))

    if n_pos == 0 or n_neg == 0:
        # Threshold-independent metrics are undefined; report NaNs but still
        # return shape info + timing.
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "precision": _safe(precision_score, y_true, flags, zero_division=0),
            "recall": _safe(recall_score, y_true, flags, zero_division=0),
            "f1": _safe(f1_score, y_true, flags, zero_division=0),
            "f1_ci_low": float("nan"),
            "f1_ci_high": float("nan"),
            "mcc": _safe(matthews_corrcoef, y_true, flags, default=0.0),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "n_total": int(len(y_true)),
            "n_positive": n_pos,
            "prevalence": prevalence,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
        }

    pr_auc_val = pr_auc(y_true, scores)
    roc_auc_val = _safe(roc_auc_score, y_true, scores)
    ap_val = _safe(average_precision_score, y_true, scores)

    f1_val = _safe(f1_score, y_true, flags, zero_division=0)
    f1_lo, f1_hi = bootstrap_f1_ci(y_true, flags, n_resamples=n_resamples_bootstrap, rng=rng)

    cm = confusion_matrix(y_true, flags, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "average_precision": ap_val,
        "precision": _safe(precision_score, y_true, flags, zero_division=0),
        "recall": _safe(recall_score, y_true, flags, zero_division=0),
        "f1": f1_val,
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "mcc": _safe(matthews_corrcoef, y_true, flags, default=0.0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_total": int(len(y_true)),
        "n_positive": n_pos,
        "prevalence": prevalence,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
    }
