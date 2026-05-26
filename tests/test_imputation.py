# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for spotanomaly2.domain.imputation (PSM + single-gap fill)."""

import logging

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain import imputation

# ----- identify_missing_data_gaps_with_count ---------------------------------


def test_identify_gaps_empty_series():
    s = pd.Series([], dtype=float)
    assert imputation.identify_missing_data_gaps_with_count(s) == []


def test_identify_gaps_no_nans():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert imputation.identify_missing_data_gaps_with_count(s) == []


def test_identify_gaps_all_nan():
    idx = pd.date_range("2025-01-01", periods=5, freq="5min")
    s = pd.Series([np.nan] * 5, index=idx)
    gaps = imputation.identify_missing_data_gaps_with_count(s)
    assert len(gaps) == 1
    assert gaps[0][2] == 5
    assert gaps[0][0] == idx[0]
    assert gaps[0][1] == idx[-1]


def test_identify_gaps_multiple_disjoint():
    idx = pd.date_range("2025-01-01", periods=10, freq="5min")
    values = [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, np.nan, np.nan]
    s = pd.Series(values, index=idx)
    gaps = imputation.identify_missing_data_gaps_with_count(s)
    # Gap 1: idx 1-2 (count 2), gap 2: idx 5 (count 1), gap 3: idx 8-9 (count 2)
    assert len(gaps) == 3
    assert gaps[0][2] == 2
    assert gaps[1][2] == 1
    assert gaps[2][2] == 2
    # Last gap extends to the end
    assert gaps[2][1] == idx[-1]


def test_identify_gaps_dataframe_input():
    idx = pd.date_range("2025-01-01", periods=5, freq="5min")
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=idx)
    gaps = imputation.identify_missing_data_gaps_with_count(df)
    assert len(gaps) == 2
    assert all(g[2] == 1 for g in gaps)


# ----- fill_missing_with_mean ------------------------------------------------


def test_fill_missing_with_mean_single_gap():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    out = imputation.fill_missing_with_mean(s)
    assert out.iloc[2] == pytest.approx(3.0)
    assert not out.isna().any()


def test_fill_missing_with_mean_only_fills_length_1_gaps():
    # Length-2 gap is NOT filled (only single-NaN holes get filled)
    s = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])
    out = imputation.fill_missing_with_mean(s)
    assert out.isna().sum() == 2


def test_fill_missing_with_mean_no_nan_unchanged():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = imputation.fill_missing_with_mean(s)
    pd.testing.assert_series_equal(s, out)


def test_fill_missing_with_mean_edge_nan_not_filled():
    # NaN at the start cannot be filled (no left neighbour)
    s = pd.Series([np.nan, 2.0, 3.0, 4.0])
    out = imputation.fill_missing_with_mean(s)
    assert pd.isna(out.iloc[0])


def test_fill_missing_with_mean_requires_series():
    with pytest.raises(ValueError, match="Pandas Series"):
        imputation.fill_missing_with_mean(np.array([1.0, np.nan, 3.0]))


def test_fill_missing_with_mean_multiple_single_gaps():
    s = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0])
    out = imputation.fill_missing_with_mean(s)
    assert out.iloc[1] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(5.0)


# ----- subsequence_imputation (PSM) ------------------------------------------


def _make_periodic_series(n=100, period=10, freq="5min", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    base = np.sin(2 * np.pi * np.arange(n) / period) * 10 + 50
    base += rng.standard_normal(n) * 0.01  # tiny noise so windows differ
    return pd.Series(base, index=idx)


def test_subsequence_imputation_mid_gap():
    s = _make_periodic_series()
    # Introduce a gap in the middle of length 3
    s_with_gap = s.copy()
    s_with_gap.iloc[40:43] = np.nan

    result = imputation.subsequence_imputation(s_with_gap)
    # Mid-gap should be filled (left + right context available)
    assert not result.iloc[40:43].isna().any()


def test_subsequence_imputation_gap_at_start():
    s = _make_periodic_series()
    # Gap at the very beginning — no left match possible
    s_with_gap = s.copy()
    s_with_gap.iloc[0:3] = np.nan

    result = imputation.subsequence_imputation(s_with_gap)
    # Right side should still fill it
    assert not result.iloc[0:3].isna().any()


def test_subsequence_imputation_gap_at_end():
    s = _make_periodic_series()
    s_with_gap = s.copy()
    s_with_gap.iloc[-3:] = np.nan

    result = imputation.subsequence_imputation(s_with_gap)
    # Left side should fill it
    assert not result.iloc[-3:].isna().any()


def test_subsequence_imputation_requires_series():
    with pytest.raises(ValueError, match="Pandas Series"):
        imputation.subsequence_imputation(np.array([1.0, 2.0, 3.0]))


def test_subsequence_imputation_requires_datetime_index():
    s = pd.Series([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        imputation.subsequence_imputation(s)


def test_subsequence_imputation_requires_frequency():
    idx = pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-05"])
    s = pd.Series([1.0, np.nan, 3.0], index=idx)
    with pytest.raises(ValueError, match="frequency"):
        imputation.subsequence_imputation(s)


def test_subsequence_imputation_logger_warns_when_unfillable(caplog):
    # All-NaN series cannot be filled by PSM — should log a warning
    idx = pd.date_range("2025-01-01", periods=10, freq="5min")
    s = pd.Series([np.nan] * 10, index=idx)
    logger = logging.getLogger("psm_test")
    with caplog.at_level(logging.WARNING, logger="psm_test"):
        result = imputation.subsequence_imputation(s, logger=logger)
    # Cannot be imputed, still NaN
    assert result.isna().all()
    assert any("PSM" in rec.message or "could not fill" in rec.message for rec in caplog.records)


def test_subsequence_imputation_no_gaps_unchanged():
    s = _make_periodic_series()
    result = imputation.subsequence_imputation(s)
    pd.testing.assert_series_equal(result, s)


# ----- vectorized_subsequence_distances --------------------------------------


def test_vectorized_subsequence_distances_basic():
    idx = pd.date_range("2025-01-01", periods=10, freq="5min")
    s = pd.Series(np.arange(10, dtype=float), index=idx)
    sub = pd.Series([0.0, 1.0, 2.0])
    df = imputation.vectorized_subsequence_distances(s, sub)
    assert "distance" in df.columns
    assert "start_idx" in df.columns
    assert "end_idx" in df.columns
    # First window matches the subsequence exactly
    assert df["distance"].iloc[0] == pytest.approx(0.0)


def test_vectorized_subsequence_distances_subseq_too_long():
    idx = pd.date_range("2025-01-01", periods=3, freq="5min")
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    sub = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="longer"):
        imputation.vectorized_subsequence_distances(s, sub)


def test_vectorized_subsequence_distances_requires_datetime_index():
    s = pd.Series([1.0, 2.0, 3.0])
    sub = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        imputation.vectorized_subsequence_distances(s, sub)


# ----- series_mean -----------------------------------------------------------


def test_series_mean():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    out = imputation.series_mean(a, b)
    np.testing.assert_array_almost_equal(out, np.array([2.0, 3.0, 4.0]))
