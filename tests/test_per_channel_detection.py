"""Tests for per-channel anomaly detection in AnomalyDetector."""

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.anomaly_detector import AnomalyDetector


@pytest.fixture
def detector_config(sample_config):
    """Config with per-channel detection enabled."""
    cfg = sample_config.copy()
    cfg["detect"] = {
        **cfg["detect"],
        "per_channel": {
            "enabled": True,
            "high_quantile": 0.99,
            "min_channels": 1,
        },
    }
    return cfg


@pytest.fixture
def residual_data():
    """Synthetic fit/eval DataFrames mimicking residual inputs.

    Returns (fit_true, fit_pred, eval_true, eval_pred) where:
      - fit window has 200 samples of normal residuals
      - eval window has 50 samples, with a spike injected into channel_b
    """
    rng = np.random.default_rng(42)
    n_fit = 200
    n_eval = 50
    channels = ["channel_a", "channel_b", "channel_c"]
    fit_idx = pd.date_range("2025-01-01", periods=n_fit, freq="5min", tz="UTC")
    eval_idx = pd.date_range(fit_idx[-1] + pd.Timedelta("5min"), periods=n_eval, freq="5min", tz="UTC")

    # Normal predictions: small residuals
    fit_true = pd.DataFrame(rng.standard_normal((n_fit, 3)) * 0.5 + 10, index=fit_idx, columns=channels)
    fit_pred = fit_true + rng.standard_normal((n_fit, 3)) * 0.1

    eval_true = pd.DataFrame(rng.standard_normal((n_eval, 3)) * 0.5 + 10, index=eval_idx, columns=channels)
    eval_pred = eval_true + rng.standard_normal((n_eval, 3)) * 0.1

    # Inject a strong spike into channel_b at indices 20-22
    eval_true.iloc[20:23, 1] += 10.0  # channel_b gets a huge deviation

    return fit_true, fit_pred, eval_true, eval_pred


class TestPerChannelDetection:
    """Tests for _detect_per_channel."""

    def test_basic_structure(self, detector_config, residual_data):
        """Result dict has expected keys with correct shapes."""
        detector = AnomalyDetector(detector_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data

        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        assert set(result.keys()) == {"scores", "flags", "thresholds", "flags_combined"}
        assert result["scores"].shape == eval_true.shape
        assert result["flags"].shape == eval_true.shape
        assert result["thresholds"].shape == (1, len(eval_true.columns))
        assert result["flags_combined"].shape == (len(eval_true), 1)
        assert "per_channel_anomaly_flag" in result["flags_combined"].columns

    def test_spike_detected_in_correct_channel(self, detector_config, residual_data):
        """The injected spike in channel_b should be flagged."""
        detector = AnomalyDetector(detector_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data

        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        # channel_b should have flags at the spike timestamps
        channel_b_flags = result["flags"]["channel_b"]
        assert channel_b_flags.iloc[20:23].sum() >= 2, "Spike in channel_b should be flagged"

        # channel_a and channel_c should have no or very few flags
        assert result["flags"]["channel_a"].sum() <= 2
        assert result["flags"]["channel_c"].sum() <= 2

    def test_combined_flag_fires(self, detector_config, residual_data):
        """Combined per-channel flag should fire when any single channel exceeds threshold."""
        detector = AnomalyDetector(detector_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data

        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        combined = result["flags_combined"]["per_channel_anomaly_flag"]
        assert combined.sum() >= 2, "Combined flag should fire for channel_b spike"

    def test_min_channels_filters(self, detector_config, residual_data):
        """With min_channels=2 the spike in only one channel should not trigger combined flag."""
        detector_config["detect"]["per_channel"]["min_channels"] = 2
        detector = AnomalyDetector(detector_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data

        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        # Only channel_b spikes — with min_channels=2, combined should rarely fire
        combined = result["flags_combined"]["per_channel_anomaly_flag"]
        assert combined.sum() <= 1, "min_channels=2 should suppress single-channel anomaly"

    def test_thresholds_are_finite(self, detector_config, residual_data):
        """All thresholds should be finite positive numbers."""
        detector = AnomalyDetector(detector_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data

        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        for col in result["thresholds"].columns:
            val = result["thresholds"][col].iloc[0]
            assert np.isfinite(val), f"Threshold for {col} should be finite"
            assert val > 0, f"Threshold for {col} should be positive"

    def test_flags_are_binary(self, detector_config, residual_data):
        """All flag values should be 0 or 1."""
        detector = AnomalyDetector(detector_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data

        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        for col in result["flags"].columns:
            unique_vals = set(result["flags"][col].unique())
            assert unique_vals <= {0, 1}, f"Flags for {col} should be binary"

        unique_combined = set(result["flags_combined"]["per_channel_anomaly_flag"].unique())
        assert unique_combined <= {0, 1}

    def test_distributed_weak_anomaly_not_flagged(self, detector_config):
        """Multiple channels with small residuals should NOT trigger per-channel flags.

        This is the core scenario: the combined KMeans detector might fire because
        the sum of small deviations is large, but per-channel detection should not
        because no single channel strongly deviates.
        """
        rng = np.random.default_rng(123)
        n_fit = 300
        n_eval = 50
        channels = ["ch_a", "ch_b", "ch_c", "ch_d", "ch_e"]
        fit_idx = pd.date_range("2025-01-01", periods=n_fit, freq="5min", tz="UTC")
        eval_idx = pd.date_range(fit_idx[-1] + pd.Timedelta("5min"), periods=n_eval, freq="5min", tz="UTC")

        # Normal training data
        fit_true = pd.DataFrame(rng.standard_normal((n_fit, 5)) + 10, index=fit_idx, columns=channels)
        fit_pred = fit_true + rng.standard_normal((n_fit, 5)) * 0.2

        # Eval: all channels have slightly elevated residuals at index 25
        eval_true = pd.DataFrame(rng.standard_normal((n_eval, 5)) + 10, index=eval_idx, columns=channels)
        eval_pred = eval_true + rng.standard_normal((n_eval, 5)) * 0.2

        # Add a small bump to ALL channels at index 25.
        # 0.3 sigma per channel — comfortably below any individual threshold
        # but the Euclidean norm across 5 channels would be ~0.67 which could
        # push a combined scorer over its threshold.
        eval_true.iloc[25, :] += 0.3

        detector = AnomalyDetector(detector_config)
        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        # Per-channel: no individual channel should flag this small distributed bump
        for col in channels:
            assert result["flags"][col].iloc[25] == 0, (
                f"{col} should NOT flag a small distributed anomaly (residual ~0.3 sigma)"
            )

    def test_single_strong_channel_flagged(self, detector_config):
        """A single channel with a very strong spike should be flagged even if others are quiet."""
        rng = np.random.default_rng(99)
        n_fit = 300
        n_eval = 50
        channels = ["ch_a", "ch_b", "ch_c"]
        fit_idx = pd.date_range("2025-01-01", periods=n_fit, freq="5min", tz="UTC")
        eval_idx = pd.date_range(fit_idx[-1] + pd.Timedelta("5min"), periods=n_eval, freq="5min", tz="UTC")

        fit_true = pd.DataFrame(rng.standard_normal((n_fit, 3)) + 10, index=fit_idx, columns=channels)
        fit_pred = fit_true + rng.standard_normal((n_fit, 3)) * 0.1

        eval_true = pd.DataFrame(rng.standard_normal((n_eval, 3)) + 10, index=eval_idx, columns=channels)
        eval_pred = eval_true + rng.standard_normal((n_eval, 3)) * 0.1

        # Strong spike in only ch_a
        eval_true.iloc[30, 0] += 20.0

        detector = AnomalyDetector(detector_config)
        result = detector._detect_per_channel("test", fit_true, fit_pred, eval_true, eval_pred)

        assert result["flags"]["ch_a"].iloc[30] == 1, "Strong spike in ch_a should be flagged"
        assert result["flags_combined"]["per_channel_anomaly_flag"].iloc[30] == 1
