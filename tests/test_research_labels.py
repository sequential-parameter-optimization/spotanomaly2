# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotanomaly2.research.labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.research.labels import labels_from_known, merge_labels


@pytest.fixture
def index_utc() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=288, freq="5min", tz="UTC")


def test_labels_from_known_marks_window(index_utc: pd.DatetimeIndex) -> None:
    known = [
        {
            "start": "2026-01-01T02:00:00+00:00",
            "end": "2026-01-01T03:00:00+00:00",
        }
    ]
    labels = labels_from_known(index_utc, known)
    n_marked = int(labels.sum())
    # 1 hour at 5-min freq, inclusive endpoints → 13 timestamps (00:00..00:60)
    assert n_marked == 13


def test_labels_from_known_with_buffer(index_utc: pd.DatetimeIndex) -> None:
    known = [
        {
            "start": "2026-01-01T02:00:00+00:00",
            "end": "2026-01-01T02:00:00+00:00",
        }
    ]
    labels = labels_from_known(index_utc, known, buffer="30min")
    # 30 min on each side at 5 min freq → 7 timestamps
    assert int(labels.sum()) == 13


def test_labels_from_known_naive_to_tz(index_utc: pd.DatetimeIndex) -> None:
    known = [
        {
            "start": "2026-01-01T02:00:00",
            "end": "2026-01-01T03:00:00",
        }
    ]
    labels = labels_from_known(index_utc, known)
    assert int(labels.sum()) > 0


def test_labels_from_known_empty_returns_zeros(index_utc: pd.DatetimeIndex) -> None:
    labels = labels_from_known(index_utc, None)
    assert int(labels.sum()) == 0
    assert len(labels) == len(index_utc)


def test_merge_labels_or(index_utc: pd.DatetimeIndex) -> None:
    a = pd.Series(np.zeros(len(index_utc), dtype=np.int8), index=index_utc)
    b = pd.Series(np.zeros(len(index_utc), dtype=np.int8), index=index_utc)
    a.iloc[10] = 1
    b.iloc[20] = 1
    merged = merge_labels(a, b)
    assert int(merged.sum()) == 2
    assert merged.iloc[10] == 1
    assert merged.iloc[20] == 1


def test_merge_labels_rejects_misaligned() -> None:
    a = pd.Series([0, 1, 0], index=pd.date_range("2026-01-01", periods=3))
    b = pd.Series([0, 0, 1], index=pd.date_range("2026-01-02", periods=3))
    with pytest.raises(ValueError):
        merge_labels(a, b)
