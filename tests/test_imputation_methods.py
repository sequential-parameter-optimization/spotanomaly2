# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for spotanomaly2.domain.imputation_methods (strategy registry)."""

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.imputation import (
    IMPUTATION_METHODS,
    BackwardFillImputation,
    ForwardFillImputation,
    ImputationMethod,
    IterativeImputation,
    KNNSklearnImputation,
    KNNTemporalImputation,
    LinearInterpolationImputation,
    MeanNeighborImputation,
    RollingMeanImputation,
    SeasonalImputation,
    SplineInterpolationImputation,
    get_imputation_method,
    impute_dataframe,
    impute_series,
    impute_series_with_weight,
)


def _make_series_with_nans(n=300, missing_ratio=0.05, seed=42):
    """Synthetic seasonal series with a few NaNs sprinkled in."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    t = np.arange(n)
    values = 50.0 + 10.0 * np.sin(2 * np.pi * t / 50) + rng.standard_normal(n) * 0.5
    series = pd.Series(values, index=idx)
    n_missing = int(missing_ratio * n)
    missing_idx = rng.choice(n, size=n_missing, replace=False)
    series.iloc[missing_idx] = np.nan
    return series


# ----- Base + happy-path coverage of every registered method -----------------


@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("mean", {}),
        ("forward_fill", {}),
        ("backward_fill", {}),
        ("linear_interpolation", {}),
        ("spline_interpolation", {}),
        ("knn_temporal", {"n_neighbors": 3}),
        ("seasonal", {"period": 50}),
        ("rolling_mean", {"window": 5}),
        ("iterative", {"max_iter": 3}),
        ("knn_sklearn", {"n_neighbors": 3}),
    ],
)
def test_method_fills_nans(method_name, kwargs):
    series = _make_series_with_nans()
    n_missing_before = int(series.isna().sum())
    assert n_missing_before > 0

    method = get_imputation_method(method_name, **kwargs)
    out = method.impute(series)

    assert isinstance(out, pd.Series)
    assert out.shape == series.shape
    # Output must have no NaNs (since the input has at least one non-NaN)
    assert int(out.isna().sum()) == 0


def test_base_class_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        ImputationMethod().impute(pd.Series([1.0]))


# ----- get_imputation_method / registry --------------------------------------


def test_get_imputation_method_returns_instance():
    inst = get_imputation_method("linear_interpolation")
    assert isinstance(inst, LinearInterpolationImputation)


def test_get_imputation_method_unknown_raises():
    with pytest.raises(ValueError, match="Unknown imputation method"):
        get_imputation_method("does_not_exist")


def test_get_imputation_method_passes_kwargs():
    inst = get_imputation_method("knn_temporal", n_neighbors=7)
    assert isinstance(inst, KNNTemporalImputation)
    assert inst.n_neighbors == 7


def test_get_imputation_method_psm_fills_multi_point_gap():
    from spotanomaly2.domain.imputation import PSMImputation

    idx = pd.date_range("2025-01-01", periods=100, freq="5min")
    base = np.sin(2 * np.pi * np.arange(100) / 10) * 10 + 50
    base += np.random.default_rng(0).standard_normal(100) * 0.01  # tiny noise so windows differ
    series = pd.Series(base, index=idx)
    series.iloc[40:43] = np.nan  # 3-point gap (PSM territory, not single-gap)

    method = get_imputation_method("psm")

    assert isinstance(method, PSMImputation)
    out = method.impute(series)
    assert not out.iloc[40:43].isna().any()


def test_registry_keys_match_classes():
    assert IMPUTATION_METHODS["mean"] is MeanNeighborImputation
    assert IMPUTATION_METHODS["forward_fill"] is ForwardFillImputation
    assert IMPUTATION_METHODS["backward_fill"] is BackwardFillImputation
    assert IMPUTATION_METHODS["linear_interpolation"] is LinearInterpolationImputation
    assert IMPUTATION_METHODS["spline_interpolation"] is SplineInterpolationImputation
    assert IMPUTATION_METHODS["knn_temporal"] is KNNTemporalImputation
    assert IMPUTATION_METHODS["seasonal"] is SeasonalImputation
    assert IMPUTATION_METHODS["rolling_mean"] is RollingMeanImputation
    assert IMPUTATION_METHODS["iterative"] is IterativeImputation
    assert IMPUTATION_METHODS["knn_sklearn"] is KNNSklearnImputation


# ----- impute_series / impute_series_with_weight / impute_dataframe ---------


def test_impute_series_convenience():
    series = _make_series_with_nans()
    out = impute_series(series, method="linear_interpolation")
    assert isinstance(out, pd.Series)
    assert out.isna().sum() == 0


def test_impute_series_with_weight_returns_pair():
    series = _make_series_with_nans()
    imputed, weight = impute_series_with_weight(series, method="linear_interpolation")
    assert imputed.shape == series.shape
    assert weight.shape == series.shape
    # weight=1 where original was non-NaN, 0 where imputed
    np.testing.assert_array_equal(weight.values, (~series.isna()).astype(int).values)
    assert imputed.isna().sum() == 0


def test_impute_dataframe_default_numeric():
    series_a = _make_series_with_nans(seed=1)
    series_b = _make_series_with_nans(seed=2)
    df = pd.DataFrame({"a": series_a, "b": series_b, "label": ["x"] * len(series_a)})
    out = impute_dataframe(df, method="linear_interpolation")
    # Only numeric columns are imputed
    assert out["a"].isna().sum() == 0
    assert out["b"].isna().sum() == 0
    # Non-numeric column unchanged
    assert (out["label"] == "x").all()


def test_impute_dataframe_explicit_columns():
    series_a = _make_series_with_nans(seed=1)
    series_b = _make_series_with_nans(seed=2)
    df = pd.DataFrame({"a": series_a, "b": series_b})
    out = impute_dataframe(df, method="linear_interpolation", columns=["a"])
    assert out["a"].isna().sum() == 0
    # 'b' was not requested
    assert out["b"].isna().sum() == df["b"].isna().sum()


# ----- Specific method properties --------------------------------------------


def test_mean_neighbor_fills_isolated_singles():
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    out = MeanNeighborImputation().impute(series)
    assert out.iloc[2] == pytest.approx(3.0)
    assert out.isna().sum() == 0


def test_forward_fill_propagates_forward():
    series = pd.Series([1.0, np.nan, np.nan, 4.0])
    out = ForwardFillImputation().impute(series)
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 1.0
    assert out.iloc[3] == 4.0


def test_backward_fill_propagates_backward():
    series = pd.Series([1.0, np.nan, np.nan, 4.0])
    out = BackwardFillImputation().impute(series)
    assert out.iloc[1] == 4.0
    assert out.iloc[2] == 4.0


def test_linear_interpolation_value():
    series = pd.Series([0.0, np.nan, np.nan, 3.0])
    out = LinearInterpolationImputation().impute(series)
    # Linear: 0, 1, 2, 3
    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(2.0)


def test_spline_interpolation_fills():
    series = _make_series_with_nans()
    out = SplineInterpolationImputation(order=2).impute(series)
    assert out.isna().sum() == 0


def test_knn_temporal_with_no_valid_values_returns_unchanged():
    # Series entirely NaN -- nothing to use as neighbour
    series = pd.Series([np.nan, np.nan, np.nan])
    out = KNNTemporalImputation(n_neighbors=3).impute(series)
    assert out.isna().all()


def test_seasonal_with_no_period_falls_back_to_neighbors_then_mean():
    # Period larger than series → seasonal lookups always fail → falls back to neighbours, then mean
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    out = SeasonalImputation(period=10000).impute(series)
    # Neighbour fallback fills index 2 with (2 + 4) / 2 = 3.0
    assert out.iloc[2] == pytest.approx(3.0)


def test_rolling_mean_window():
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    out = RollingMeanImputation(window=2).impute(series)
    # Window of 2 around idx 2 looks at idx 0..4 → mean of {1,2,4,5} = 3.0
    assert out.iloc[2] == pytest.approx(3.0)


# ----- sklearn-backed methods (available=True path; mock the unavailable) ----


def test_iterative_imputation_when_unavailable_raises(monkeypatch):
    impu = IterativeImputation()
    monkeypatch.setattr(impu, "available", False)
    series = _make_series_with_nans()
    with pytest.raises(ImportError, match="scikit-learn"):
        impu.impute(series)


def test_knn_sklearn_imputation_when_unavailable_raises(monkeypatch):
    impu = KNNSklearnImputation()
    monkeypatch.setattr(impu, "available", False)
    series = _make_series_with_nans()
    with pytest.raises(ImportError, match="scikit-learn"):
        impu.impute(series)
