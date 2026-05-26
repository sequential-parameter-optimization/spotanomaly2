# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from spotanomaly2.research.metrics import bootstrap_f1_ci, compute_metrics, pr_auc


def test_pr_auc_perfect_separation() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert pr_auc(y_true, scores) == pytest.approx(1.0, abs=1e-6)


def test_pr_auc_random_scores_equal_prevalence_baseline() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=10000)
    scores = rng.random(10000)
    val = pr_auc(y_true, scores)
    # With random scores PR-AUC ≈ prevalence (~0.5 here)
    assert 0.45 <= val <= 0.55


def test_pr_auc_matches_sklearn_average_precision_within_eps() -> None:
    rng = np.random.default_rng(1)
    y_true = (rng.random(500) < 0.05).astype(int)
    scores = rng.random(500) + 0.5 * y_true
    # AP and trapezoidal PR-AUC are not identical but should be in the
    # same ballpark on a smooth synthetic example.
    assert pr_auc(y_true, scores) == pytest.approx(
        average_precision_score(y_true, scores), abs=0.1
    )


def test_pr_auc_undefined_when_one_class() -> None:
    y_true = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    assert math.isnan(pr_auc(y_true, scores))


def test_bootstrap_f1_ci_returns_interval() -> None:
    rng = np.random.default_rng(42)
    y_true = (rng.random(200) < 0.1).astype(int)
    flags = y_true.copy()
    # Inject some noise so F1 isn't perfect.
    flip = rng.integers(0, 200, size=20)
    flags[flip] = 1 - flags[flip]
    lo, hi = bootstrap_f1_ci(y_true, flags, n_resamples=100, rng=np.random.default_rng(0))
    assert 0.0 <= lo <= hi <= 1.0


def test_compute_metrics_full_bundle() -> None:
    rng = np.random.default_rng(2)
    y_true = (rng.random(1000) < 0.05).astype(int)
    scores = rng.random(1000) + 0.6 * y_true
    flags = (scores > 0.7).astype(int)
    out = compute_metrics(
        y_true,
        scores,
        flags,
        fit_seconds=0.5,
        score_seconds=0.1,
        n_resamples_bootstrap=50,
    )
    assert set(out.keys()) >= {
        "pr_auc",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "f1_ci_low",
        "f1_ci_high",
        "mcc",
        "tp",
        "fp",
        "fn",
        "tn",
        "n_total",
        "n_positive",
        "prevalence",
        "fit_seconds",
        "score_seconds",
    }
    assert out["fit_seconds"] == 0.5
    assert 0.0 <= out["pr_auc"] <= 1.0
    assert out["n_total"] == 1000
    assert out["n_positive"] == int(y_true.sum())


def test_compute_metrics_handles_all_zeros_y_true() -> None:
    out = compute_metrics(
        y_true=np.zeros(10, dtype=int),
        scores=np.linspace(0, 1, 10),
        flags=np.zeros(10, dtype=int),
    )
    assert math.isnan(out["pr_auc"])
    assert out["n_positive"] == 0
    assert out["tp"] == 0
