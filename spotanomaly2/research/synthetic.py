# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synthetic anomaly injection for benchmarking.

Three anomaly archetypes from the time-series literature:

- **Point**: a single timestamp's value is replaced by an extreme spike.
- **Contextual**: a single timestamp's value is shifted by a magnitude that
  is moderate in absolute terms but extreme relative to the surrounding
  window — undetectable without context.
- **Collective**: a contiguous run of timestamps is shifted by a uniform
  offset, simulating a sustained level-shift / fault.

All injectors operate on a copy of the input DataFrame and return both the
mutated frame and a binary label vector aligned with the frame's index.
Injection sites are chosen uniformly at random from the eligible index
range, with a minimum stride between events to keep them disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Injector primitives
# ---------------------------------------------------------------------------

# Default magnitudes are expressed in standard deviations of the column. They
# are deliberately large enough that a *competent* scorer should detect them.
# The benchmark sweeps prevalence, not magnitude, so these are fixed defaults.
DEFAULT_POINT_SIGMA = 8.0
DEFAULT_CONTEXTUAL_SIGMA = 4.0
DEFAULT_COLLECTIVE_SIGMA = 3.0
DEFAULT_COLLECTIVE_LEN = 12  # 1 hour at 5-min freq
DEFAULT_CONTEXTUAL_WINDOW = 24


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of an injection step.

    Attributes:
        df: Frame with injected values.
        labels: Binary Series aligned with df.index (1 = injected anomaly).
        sites: List of (timestamp, anomaly_type) tuples for diagnostics.
    """

    df: pd.DataFrame
    labels: pd.Series
    sites: list[tuple[pd.Timestamp, str]]


def _sample_disjoint_positions(
    n_total: int,
    n_events: int,
    min_stride: int,
    rng: np.random.Generator,
    margin: int = 0,
) -> np.ndarray:
    """Sample n_events positions from [margin, n_total - margin) with min_stride apart.

    Falls back to fewer events if the index isn't long enough to satisfy the
    stride constraint. Caller should clip n_events to feasible upper bound
    before calling.
    """
    lo = margin
    hi = n_total - margin
    if hi - lo <= 0 or n_events <= 0:
        return np.array([], dtype=int)

    candidates: list[int] = []
    attempts = 0
    max_attempts = max(50, n_events * 20)
    while len(candidates) < n_events and attempts < max_attempts:
        pos = int(rng.integers(lo, hi))
        if all(abs(pos - existing) >= min_stride for existing in candidates):
            candidates.append(pos)
        attempts += 1
    return np.array(sorted(candidates), dtype=int)


def inject_point(
    df: pd.DataFrame,
    n_events: int,
    rng: np.random.Generator,
    sigma: float = DEFAULT_POINT_SIGMA,
    columns: list[str] | None = None,
) -> InjectionResult:
    """Inject point anomalies (single-timestamp spikes)."""
    out = df.copy()
    target_cols = columns if columns is not None else list(df.select_dtypes(include=np.number).columns)
    labels = pd.Series(0, index=df.index, dtype=np.int8)
    sites: list[tuple[pd.Timestamp, str]] = []

    if not target_cols or n_events <= 0:
        return InjectionResult(df=out, labels=labels, sites=sites)

    positions = _sample_disjoint_positions(len(df), n_events, min_stride=4, rng=rng)
    for pos in positions:
        col = target_cols[int(rng.integers(0, len(target_cols)))]
        std = float(df[col].std()) or 1.0
        sign = 1.0 if rng.random() > 0.5 else -1.0
        out.iloc[pos, out.columns.get_loc(col)] = float(df[col].iloc[pos]) + sign * sigma * std
        labels.iloc[pos] = 1
        sites.append((df.index[pos], "point"))

    return InjectionResult(df=out, labels=labels, sites=sites)


def inject_contextual(
    df: pd.DataFrame,
    n_events: int,
    rng: np.random.Generator,
    sigma: float = DEFAULT_CONTEXTUAL_SIGMA,
    window: int = DEFAULT_CONTEXTUAL_WINDOW,
    columns: list[str] | None = None,
) -> InjectionResult:
    """Inject contextual anomalies — moderate global, extreme local."""
    out = df.copy()
    target_cols = columns if columns is not None else list(df.select_dtypes(include=np.number).columns)
    labels = pd.Series(0, index=df.index, dtype=np.int8)
    sites: list[tuple[pd.Timestamp, str]] = []

    if not target_cols or n_events <= 0:
        return InjectionResult(df=out, labels=labels, sites=sites)

    positions = _sample_disjoint_positions(
        len(df), n_events, min_stride=window + 2, rng=rng, margin=window
    )
    for pos in positions:
        col = target_cols[int(rng.integers(0, len(target_cols)))]
        local = df[col].iloc[max(0, pos - window) : pos + window + 1]
        local_std = float(local.std()) or float(df[col].std()) or 1.0
        local_mean = float(local.mean())
        sign = 1.0 if rng.random() > 0.5 else -1.0
        out.iloc[pos, out.columns.get_loc(col)] = local_mean + sign * sigma * local_std
        labels.iloc[pos] = 1
        sites.append((df.index[pos], "contextual"))

    return InjectionResult(df=out, labels=labels, sites=sites)


def inject_collective(
    df: pd.DataFrame,
    n_events: int,
    rng: np.random.Generator,
    sigma: float = DEFAULT_COLLECTIVE_SIGMA,
    length: int = DEFAULT_COLLECTIVE_LEN,
    columns: list[str] | None = None,
) -> InjectionResult:
    """Inject collective anomalies — contiguous level-shift over `length` rows."""
    out = df.copy()
    target_cols = columns if columns is not None else list(df.select_dtypes(include=np.number).columns)
    labels = pd.Series(0, index=df.index, dtype=np.int8)
    sites: list[tuple[pd.Timestamp, str]] = []

    if not target_cols or n_events <= 0:
        return InjectionResult(df=out, labels=labels, sites=sites)

    positions = _sample_disjoint_positions(
        len(df), n_events, min_stride=length * 2, rng=rng, margin=length
    )
    for pos in positions:
        col = target_cols[int(rng.integers(0, len(target_cols)))]
        std = float(df[col].std()) or 1.0
        sign = 1.0 if rng.random() > 0.5 else -1.0
        offset = sign * sigma * std
        end = min(pos + length, len(df))
        col_idx = out.columns.get_loc(col)
        out.iloc[pos:end, col_idx] = out.iloc[pos:end, col_idx].to_numpy() + offset
        labels.iloc[pos:end] = 1
        sites.append((df.index[pos], "collective"))

    return InjectionResult(df=out, labels=labels, sites=sites)


# ---------------------------------------------------------------------------
# Combined injector
# ---------------------------------------------------------------------------


def inject_all(
    df: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
    columns: list[str] | None = None,
) -> InjectionResult:
    """Run all three injectors back-to-back, accumulating labels.

    Reads counts, magnitudes, and seed from ``evaluate.synthetic``:

    - ``n_point``, ``n_contextual``, ``n_collective``, ``random_seed``
    - ``point_sigma``, ``contextual_sigma``, ``collective_sigma`` —
      injection magnitudes (in standard deviations of the column)
    - ``contextual_window`` — local window length for contextual anomalies
    - ``collective_length`` — number of contiguous rows per collective event

    Sigmas default to the module-level constants (deliberately strong, so
    a smoke test produces clearly visible signal); lower them in config
    for a discriminating benchmark.
    """
    synthetic_cfg = config.get("evaluate", {}).get("synthetic", {})
    if rng is None:
        seed = int(synthetic_cfg.get("random_seed", 42))
        rng = np.random.default_rng(seed)

    n_point = int(synthetic_cfg.get("n_point", 0))
    n_contextual = int(synthetic_cfg.get("n_contextual", 0))
    n_collective = int(synthetic_cfg.get("n_collective", 0))

    point_sigma = float(synthetic_cfg.get("point_sigma", DEFAULT_POINT_SIGMA))
    contextual_sigma = float(synthetic_cfg.get("contextual_sigma", DEFAULT_CONTEXTUAL_SIGMA))
    collective_sigma = float(synthetic_cfg.get("collective_sigma", DEFAULT_COLLECTIVE_SIGMA))
    contextual_window = int(synthetic_cfg.get("contextual_window", DEFAULT_CONTEXTUAL_WINDOW))
    collective_length = int(synthetic_cfg.get("collective_length", DEFAULT_COLLECTIVE_LEN))

    out = df
    labels = pd.Series(0, index=df.index, dtype=np.int8)
    sites: list[tuple[pd.Timestamp, str]] = []

    if n_point > 0:
        result = inject_point(out, n_events=n_point, rng=rng, sigma=point_sigma, columns=columns)
        out = result.df
        labels = pd.Series(
            (labels.to_numpy() | result.labels.to_numpy()).astype(np.int8), index=df.index
        )
        sites.extend(result.sites)
    if n_contextual > 0:
        result = inject_contextual(
            out,
            n_events=n_contextual,
            rng=rng,
            sigma=contextual_sigma,
            window=contextual_window,
            columns=columns,
        )
        out = result.df
        labels = pd.Series(
            (labels.to_numpy() | result.labels.to_numpy()).astype(np.int8), index=df.index
        )
        sites.extend(result.sites)
    if n_collective > 0:
        result = inject_collective(
            out,
            n_events=n_collective,
            rng=rng,
            sigma=collective_sigma,
            length=collective_length,
            columns=columns,
        )
        out = result.df
        labels = pd.Series(
            (labels.to_numpy() | result.labels.to_numpy()).astype(np.int8), index=df.index
        )
        sites.extend(result.sites)

    return InjectionResult(df=out, labels=labels, sites=sites)
