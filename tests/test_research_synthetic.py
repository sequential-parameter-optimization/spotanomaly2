# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.synthetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.research.synthetic import (
    inject_all,
    inject_collective,
    inject_contextual,
    inject_point,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=2000, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "channel_a": rng.normal(loc=10.0, scale=1.0, size=len(idx)),
            "channel_b": rng.normal(loc=0.0, scale=2.0, size=len(idx)),
        },
        index=idx,
    )


def test_inject_point_places_expected_count(clean_df: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    result = inject_point(clean_df, n_events=10, rng=rng)
    assert int(result.labels.sum()) == 10
    assert len(result.sites) == 10
    assert all(kind == "point" for _, kind in result.sites)
    # Original frame should be untouched.
    assert not clean_df.equals(result.df)


def test_inject_point_seeded_reproducible(clean_df: pd.DataFrame) -> None:
    a = inject_point(clean_df, n_events=5, rng=np.random.default_rng(123))
    b = inject_point(clean_df, n_events=5, rng=np.random.default_rng(123))
    pd.testing.assert_frame_equal(a.df, b.df)
    pd.testing.assert_series_equal(a.labels, b.labels)


def test_inject_contextual_respects_window(clean_df: pd.DataFrame) -> None:
    window = 12
    rng = np.random.default_rng(42)
    result = inject_contextual(clean_df, n_events=8, rng=rng, window=window)
    # Anomalies should not be placed in the first or last `window` rows.
    positions = np.flatnonzero(result.labels.to_numpy())
    assert positions.min() >= window
    assert positions.max() < len(clean_df) - window


def test_inject_collective_marks_contiguous_runs(clean_df: pd.DataFrame) -> None:
    length = 8
    rng = np.random.default_rng(11)
    result = inject_collective(clean_df, n_events=4, rng=rng, length=length)
    # Should label exactly 4 * length rows (no overlap thanks to min_stride).
    assert int(result.labels.sum()) == 4 * length

    # Each run is contiguous: count the number of 0->1 transitions.
    arr = result.labels.to_numpy()
    transitions = int(((arr[1:] == 1) & (arr[:-1] == 0)).sum())
    assert transitions == 4


def test_inject_all_combines_three_types(clean_df: pd.DataFrame) -> None:
    config = {
        "evaluate": {
            "synthetic": {
                "n_point": 5,
                "n_contextual": 3,
                "n_collective": 2,
                "random_seed": 99,
            }
        }
    }
    result = inject_all(clean_df, config=config)
    site_kinds = [kind for _, kind in result.sites]
    assert site_kinds.count("point") == 5
    assert site_kinds.count("contextual") == 3
    assert site_kinds.count("collective") == 2
    assert int(result.labels.sum()) >= 5 + 3 + 2


def test_inject_all_handles_zero_counts(clean_df: pd.DataFrame) -> None:
    config = {
        "evaluate": {
            "synthetic": {
                "n_point": 0,
                "n_contextual": 0,
                "n_collective": 0,
                "random_seed": 0,
            }
        }
    }
    result = inject_all(clean_df, config=config)
    assert int(result.labels.sum()) == 0
    assert result.sites == []
    pd.testing.assert_frame_equal(result.df, clean_df)
