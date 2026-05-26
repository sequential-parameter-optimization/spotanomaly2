# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.gmm_scorer.GMMAnomalyScorer."""

from __future__ import annotations

import numpy as np
import pytest

from spotanomaly2.research.gmm_scorer import GMMAnomalyScorer


def test_constructor_validates_params() -> None:
    with pytest.raises(ValueError):
        GMMAnomalyScorer(covariance_type="wrong")
    with pytest.raises(ValueError):
        GMMAnomalyScorer(selection_criterion="ic")
    with pytest.raises(ValueError):
        GMMAnomalyScorer(n_components=0)


def test_fit_score_1d_residuals() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=300)
    test = rng.normal(size=50)
    scorer = GMMAnomalyScorer(n_components=2, selection_criterion="fixed")
    scorer.fit(train)
    scores = scorer.score(test)
    assert scores.shape == (50,)
    assert np.all(np.isfinite(scores))


def test_score_before_fit_raises() -> None:
    scorer = GMMAnomalyScorer()
    with pytest.raises(ValueError):
        scorer.score(np.zeros(10))


def test_outlier_scores_higher_than_inlier() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(loc=0.0, scale=1.0, size=(500, 2))
    scorer = GMMAnomalyScorer(n_components=2, selection_criterion="fixed")
    scorer.fit(train)
    inlier = np.zeros((1, 2))
    outlier = np.array([[20.0, 20.0]])
    assert scorer.score(outlier)[0] > scorer.score(inlier)[0]


def test_bic_selects_two_components_on_two_component_mix() -> None:
    rng = np.random.default_rng(2)
    n = 800
    a = rng.normal(loc=-3.0, scale=0.5, size=(n // 2, 1))
    b = rng.normal(loc=3.0, scale=0.5, size=(n // 2, 1))
    train = np.vstack([a, b])
    rng.shuffle(train)
    scorer = GMMAnomalyScorer(n_components=6, selection_criterion="bic")
    scorer.fit(train)
    # BIC should prefer 2 components for this mixture; allow 2 or 3 for
    # numerical wiggle room (the criterion can pick a slight overfit).
    assert scorer.selected_n_components in (2, 3)
    assert scorer.selection_scores is not None
    assert 2 in scorer.selection_scores


def test_aic_branch_smoke() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(200, 1))
    scorer = GMMAnomalyScorer(n_components=4, selection_criterion="aic")
    scorer.fit(train)
    assert scorer.selected_n_components is not None
    assert scorer.selection_scores is not None
