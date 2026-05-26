# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gaussian Mixture Model anomaly scorer (sklearn-backed).

Lives in ``spotanomaly2.research`` rather than ``spotanomaly2_safe`` because
``spotanomaly2_safe`` is intentionally numpy-only — adding scikit-learn to its
dependency closure would break the safety-critical minimal-deps contract.
This scorer is therefore part of the *research* surface and reachable only
through the ``BenchmarkDetector`` subclass in ``scorer_runner.py``.

API contract matches the existing scorers in
``spotanomaly2_safe.scoring.scorers``:

- ``fit(residuals: np.ndarray) -> None``
- ``score(residuals: np.ndarray) -> np.ndarray`` of shape ``(n_samples,)``

with 1-D residuals auto-reshaped to ``(n, 1)`` and the score being the
*negative* per-sample log-likelihood under the fitted GMM (higher = more
anomalous).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_VALID_COVARIANCE = {"full", "tied", "diag", "spherical"}
_VALID_SELECTION = {"bic", "aic", "fixed"}


class GMMAnomalyScorer:
    """Anomaly scorer based on a Gaussian Mixture Model.

    Fits a GMM on standard-scaled residuals and assigns each test sample
    a score equal to ``-log p(x)`` under the mixture — high score = low
    likelihood = anomalous.

    Args:
        n_components: Number of mixture components. Used directly when
            ``selection_criterion="fixed"`` and as the upper bound of the
            search range when ``"bic"`` or ``"aic"`` is chosen.
        covariance_type: Passed to ``sklearn.mixture.GaussianMixture``;
            one of ``{"full","tied","diag","spherical"}``.
        selection_criterion: How to pick ``n_components``. ``"fixed"``
            uses the supplied value; ``"bic"`` and ``"aic"`` fit one GMM
            per ``k in [1, n_components]`` and pick the lowest score.
        random_state: Seed forwarded to ``GaussianMixture`` for
            reproducibility.
        max_iter: Inner EM iteration cap; the sklearn default of 100 is
            usually fine.
        n_init: Number of EM restarts; sklearn default is 1, we default to
            3 for stability on heavy-tailed residuals.
        reg_covar: Diagonal covariance regularisation; sklearn default
            (1e-6) is preserved.

    Attributes:
        gmm: The fitted ``GaussianMixture``. ``None`` before ``fit``.
        scaler: The fitted ``StandardScaler``.
        selected_n_components: ``n_components`` actually used after
            criterion-based selection. Equals the constructor value when
            ``selection_criterion="fixed"``.
        selection_scores: Mapping ``{k: criterion_score}`` from the search,
            or ``None`` when ``selection_criterion="fixed"``.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        selection_criterion: str = "fixed",
        random_state: int = 42,
        max_iter: int = 100,
        n_init: int = 3,
        reg_covar: float = 1e-6,
    ):
        if covariance_type not in _VALID_COVARIANCE:
            raise ValueError(
                f"covariance_type must be one of {_VALID_COVARIANCE}, got {covariance_type!r}"
            )
        if selection_criterion not in _VALID_SELECTION:
            raise ValueError(
                f"selection_criterion must be one of {_VALID_SELECTION}, got {selection_criterion!r}"
            )
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.selection_criterion = selection_criterion
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.reg_covar = float(reg_covar)

        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.selected_n_components: Optional[int] = None
        self.selection_scores: Optional[dict[int, float]] = None

    @staticmethod
    def _ensure_2d(residuals: np.ndarray) -> np.ndarray:
        if residuals.ndim == 1:
            return residuals.reshape(-1, 1)
        return residuals

    def _fit_one(self, X: np.ndarray, k: int) -> GaussianMixture:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
            reg_covar=self.reg_covar,
        )
        gmm.fit(X)
        return gmm

    def fit(self, residuals: np.ndarray) -> None:
        """Fit a GMM on training residuals; optionally select n_components."""
        X = self._ensure_2d(np.asarray(residuals, dtype=float))
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.selection_criterion == "fixed":
            self.gmm = self._fit_one(X_scaled, self.n_components)
            self.selected_n_components = self.n_components
            self.selection_scores = None
            return

        scores: dict[int, float] = {}
        best_k = 1
        best_score = np.inf
        best_gmm: Optional[GaussianMixture] = None
        for k in range(1, self.n_components + 1):
            try:
                gmm = self._fit_one(X_scaled, k)
            except (ValueError, np.linalg.LinAlgError):
                # Singular covariance for this k → skip.
                continue
            score = gmm.bic(X_scaled) if self.selection_criterion == "bic" else gmm.aic(X_scaled)
            scores[k] = float(score)
            if score < best_score:
                best_score = score
                best_k = k
                best_gmm = gmm

        if best_gmm is None:
            # Fallback: fit at the requested n_components even if criterion
            # search failed across the board.
            best_gmm = self._fit_one(X_scaled, self.n_components)
            best_k = self.n_components

        self.gmm = best_gmm
        self.selected_n_components = best_k
        self.selection_scores = scores

    def score(self, residuals: np.ndarray) -> np.ndarray:
        """Score residuals as negative per-sample log-likelihood."""
        if self.gmm is None or self.scaler is None:
            raise ValueError("Scorer must be fit before scoring")
        X = self._ensure_2d(np.asarray(residuals, dtype=float))
        X_scaled = self.scaler.transform(X)
        log_prob = self.gmm.score_samples(X_scaled)
        # Negate so higher = more anomalous, matching the other scorers'
        # convention (e.g. distance-to-centroid for KMeans).
        return -np.asarray(log_prob, dtype=float)
