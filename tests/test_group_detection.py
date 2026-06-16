"""Tests for group-based scoring in AnomalyDetector.

``detect.groups`` splits a panel's channels into groups, scores each group
independently with a Bonferroni-corrected quantile, and flags the system when
any group trips (logical OR).
"""

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.anomaly_detector import AnomalyDetector


@pytest.fixture
def grouped_config(sample_config):
    """Config with two per-panel groups defined for panel 'test'."""
    cfg = sample_config.copy()
    cfg["detect"] = {
        **cfg["detect"],
        "high_quantile": 0.99,
        "groups": {
            "test": {
                "g1": ["channel_a"],
                "g2": ["channel_b", "channel_c"],
            }
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

    fit_true = pd.DataFrame(rng.standard_normal((n_fit, 3)) * 0.5 + 10, index=fit_idx, columns=channels)
    fit_pred = fit_true + rng.standard_normal((n_fit, 3)) * 0.1

    eval_true = pd.DataFrame(rng.standard_normal((n_eval, 3)) * 0.5 + 10, index=eval_idx, columns=channels)
    eval_pred = eval_true + rng.standard_normal((n_eval, 3)) * 0.1

    # Inject a strong spike into channel_b at indices 20-22
    eval_true.iloc[20:23, 1] += 10.0

    return fit_true, fit_pred, eval_true, eval_pred


class TestBonferroniQuantile:
    """Tests for the Bonferroni quantile correction."""

    def test_single_group_is_unchanged(self):
        assert AnomalyDetector._bonferroni_quantile(0.999, 1) == 0.999

    def test_two_groups_halves_the_tail(self):
        # 0.999 tail (0.001) split across 2 groups -> 0.0005 each -> 0.9995.
        assert AnomalyDetector._bonferroni_quantile(0.999, 2) == pytest.approx(0.9995)

    def test_four_groups(self):
        assert AnomalyDetector._bonferroni_quantile(0.999, 4) == pytest.approx(0.99975)

    def test_higher_base_quantile(self):
        assert AnomalyDetector._bonferroni_quantile(0.9999, 2) == pytest.approx(0.99995)


class TestResolveGroups:
    """Tests for _resolve_groups."""

    def test_none_when_no_groups_configured(self, sample_config):
        detector = AnomalyDetector(sample_config)
        assert detector._resolve_groups("test", ["channel_a", "channel_b"]) is None

    def test_none_when_panel_not_in_groups(self, grouped_config):
        detector = AnomalyDetector(grouped_config)
        # Panel "999" has no entry under detect.groups.
        assert detector._resolve_groups("999", ["channel_a"]) is None

    def test_intersects_with_available_columns(self, grouped_config):
        detector = AnomalyDetector(grouped_config)
        resolved = detector._resolve_groups("test", ["channel_a", "channel_b", "channel_c"])
        assert resolved == {
            "g1": {"columns": ["channel_a"], "quantile": None},
            "g2": {"columns": ["channel_b", "channel_c"], "quantile": None},
        }

    def test_drops_missing_columns_and_warns(self, grouped_config):
        detector = AnomalyDetector(grouped_config)
        # channel_c absent from data -> g2 keeps only channel_b.
        resolved = detector._resolve_groups("test", ["channel_a", "channel_b"])
        assert resolved == {
            "g1": {"columns": ["channel_a"], "quantile": None},
            "g2": {"columns": ["channel_b"], "quantile": None},
        }

    def test_empty_group_skipped(self, grouped_config):
        detector = AnomalyDetector(grouped_config)
        # Only g2 columns present -> g1 has no present columns and is dropped.
        resolved = detector._resolve_groups("test", ["channel_b", "channel_c"])
        assert resolved == {"g2": {"columns": ["channel_b", "channel_c"], "quantile": None}}

    def test_dict_form_with_quantile_override(self, sample_config):
        cfg = sample_config.copy()
        cfg["detect"] = {
            **cfg["detect"],
            "groups": {
                "test": {
                    "g1": ["channel_a"],
                    "g2": {"channels": ["channel_b"], "quantile": 0.999},
                }
            },
        }
        detector = AnomalyDetector(cfg)
        resolved = detector._resolve_groups("test", ["channel_a", "channel_b"])
        assert resolved == {
            "g1": {"columns": ["channel_a"], "quantile": None},
            "g2": {"columns": ["channel_b"], "quantile": 0.999},
        }

    def test_columns_alias_accepted(self):
        cols, q = AnomalyDetector._parse_group_spec("1", "g", {"columns": ["x"], "quantile": 0.99})
        assert cols == ["x"] and q == 0.99

    def test_invalid_quantile_rejected(self):
        with pytest.raises(ValueError):
            AnomalyDetector._parse_group_spec("1", "g", {"channels": ["x"], "quantile": 1.5})

    def test_invalid_spec_rejected(self):
        with pytest.raises(ValueError):
            AnomalyDetector._parse_group_spec("1", "g", "not-a-list")


class TestGroupedScoring:
    """Tests for _score_grouped end to end."""

    def test_basic_structure(self, grouped_config, residual_data):
        detector = AnomalyDetector(grouped_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data
        groups = {
            "g1": {"columns": ["channel_a"], "quantile": None},
            "g2": {"columns": ["channel_b", "channel_c"], "quantile": None},
        }

        scores_df, flags_df = detector._score_grouped("test", groups, fit_true, fit_pred, eval_true, eval_pred)

        assert "anomaly_score" in scores_df.columns
        assert "anomaly_score_normalized" in scores_df.columns
        assert "anomaly_flag" in flags_df.columns
        # Per-group detail columns are carried through.
        assert "group_g1_normalized" in scores_df.columns
        assert "group_g2_flag" in flags_df.columns
        assert len(flags_df) == len(eval_true)

    def test_flags_are_binary(self, grouped_config, residual_data):
        detector = AnomalyDetector(grouped_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data
        groups = {
            "g1": {"columns": ["channel_a"], "quantile": None},
            "g2": {"columns": ["channel_b", "channel_c"], "quantile": None},
        }

        _scores, flags_df = detector._score_grouped("test", groups, fit_true, fit_pred, eval_true, eval_pred)
        assert set(flags_df["anomaly_flag"].unique()) <= {0, 1}

    def test_spike_in_a_group_triggers_system_flag(self, grouped_config, residual_data):
        """A spike confined to g2 (channel_b) must set the system flag via OR."""
        detector = AnomalyDetector(grouped_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data
        groups = {
            "g1": {"columns": ["channel_a"], "quantile": None},
            "g2": {"columns": ["channel_b", "channel_c"], "quantile": None},
        }

        scores_df, flags_df = detector._score_grouped("test", groups, fit_true, fit_pred, eval_true, eval_pred)

        # g2 (which holds channel_b) should flag the spike window...
        assert flags_df["group_g2_flag"].iloc[20:23].sum() >= 1
        # ...and the system flag must reflect it (OR of the groups).
        assert flags_df["anomaly_flag"].iloc[20:23].sum() >= 1
        # The system flag is the OR of the per-group flags at every timestamp.
        expected = (flags_df["group_g1_flag"] | flags_df["group_g2_flag"]).astype(int)
        pd.testing.assert_series_equal(flags_df["anomaly_flag"], expected, check_names=False)

    def test_quantile_override_reaches_scorer(self, grouped_config, residual_data):
        """A group's explicit quantile is passed through; others use Bonferroni."""
        detector = AnomalyDetector(grouped_config)
        fit_true, fit_pred, eval_true, eval_pred = residual_data
        groups = {
            "g1": {"columns": ["channel_a"], "quantile": None},  # -> Bonferroni default
            "g2": {"columns": ["channel_b", "channel_c"], "quantile": 0.95},  # -> override
        }

        seen: dict[str, float] = {}
        original = detector._score_one

        def spy(panel_id, label, *args, high_quantile, **kwargs):
            seen[label] = high_quantile
            return original(panel_id, label, *args, high_quantile=high_quantile, **kwargs)

        detector._score_one = spy
        detector._score_grouped("test", groups, fit_true, fit_pred, eval_true, eval_pred)

        # high_quantile=0.99 in grouped_config, 2 groups -> Bonferroni 0.995 for g1.
        assert seen["group 'g1'"] == pytest.approx(0.995)
        assert seen["group 'g2'"] == pytest.approx(0.95)
