# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the AnomalyDetector orchestration class.

Focused on the pieces that can be unit-tested without spinning up a real
SpotforecastTrainer / trained-model artefact:

* ``load_forecasting_model`` — missing/malformed model paths (joblib.load mocked)
* ``build_anomaly_detector`` — config translation (window_agg drop, k -> n_clusters)
* ``_calculate_test_window_default`` — pure index math
* ``_adjust_window_for_insufficient_data`` — raises on too-short series
* ``_split_unseen_scoring_data`` — leakage guard (timestamp, train_size, ratio)
* ``_validate_scoring_inputs`` — NaN / Inf detection
* ``_exclude_imputed_rows`` — weight-column-driven row drop

End-to-end ``detect_panel`` is intentionally out of scope here; it needs a
fitted forecaster artefact and belongs in integration tests.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.anomaly_detector import AnomalyDetector
from spotanomaly2.domain.exceptions import InsufficientDataException, ModelNotFoundException

# ---------------------------------------------------------------------------
# load_forecasting_model
#
# Note: joblib.load is mocked everywhere; we never write pickles in tests.


class TestLoadForecastingModel:
    def test_missing_model_raises_typed_exception(self, sample_config, tmp_path):
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        (tmp_path / "models").mkdir()
        detector = AnomalyDetector(sample_config)
        with pytest.raises(ModelNotFoundException):
            detector.load_forecasting_model("1")

    def test_missing_models_dir_raises_typed_exception(self, sample_config, tmp_path):
        sample_config["paths"]["models_dir"] = str(tmp_path / "absent_models")
        detector = AnomalyDetector(sample_config)
        with pytest.raises(ModelNotFoundException):
            detector.load_forecasting_model("1")

    def test_explicit_timestamp_missing_raises(self, sample_config, tmp_path):
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        sample_config["detect"]["model_timestamp"] = "20240101_000000"
        (tmp_path / "models").mkdir()
        detector = AnomalyDetector(sample_config)
        with pytest.raises(ModelNotFoundException):
            detector.load_forecasting_model("1")

    def test_malformed_payload_raises(self, sample_config, tmp_path):
        """A loaded payload without ``model_type`` must be rejected."""
        models_dir = tmp_path / "models" / "20251231_120000"
        models_dir.mkdir(parents=True)
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        # Create the file so find_latest_model succeeds, then mock joblib.load.
        (models_dir / "fc_model_panel_1.pkl").write_bytes(b"placeholder")
        detector = AnomalyDetector(sample_config)
        with patch(
            "spotanomaly2.domain.anomaly_detector.joblib.load",
            return_value={"unrelated": "payload"},
        ):
            with pytest.raises(ModelNotFoundException, match="Unknown format"):
                detector.load_forecasting_model("1")

    def test_well_formed_payload_returns_dict(self, sample_config, tmp_path):
        models_dir = tmp_path / "models" / "20251231_120000"
        models_dir.mkdir(parents=True)
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        (models_dir / "fc_model_panel_1.pkl").write_bytes(b"placeholder")
        detector = AnomalyDetector(sample_config)
        payload = {"model_type": "LightGBM", "fake_field": 1}
        with patch(
            "spotanomaly2.domain.anomaly_detector.joblib.load",
            return_value=payload,
        ):
            loaded = detector.load_forecasting_model("1")
        assert loaded["model_type"] == "LightGBM"
        assert loaded["fake_field"] == 1

    def test_resolves_to_explicit_timestamp_when_provided(self, sample_config, tmp_path):
        models_dir_old = tmp_path / "models" / "20240101_000000"
        models_dir_new = tmp_path / "models" / "20251231_120000"
        models_dir_old.mkdir(parents=True)
        models_dir_new.mkdir(parents=True)
        (models_dir_old / "fc_model_panel_1.pkl").write_bytes(b"old")
        (models_dir_new / "fc_model_panel_1.pkl").write_bytes(b"new")
        sample_config["paths"]["models_dir"] = str(tmp_path / "models")
        sample_config["detect"]["model_timestamp"] = "20240101_000000"
        detector = AnomalyDetector(sample_config)
        with patch(
            "spotanomaly2.domain.anomaly_detector.joblib.load",
            return_value={"model_type": "LightGBM"},
        ) as mock_load:
            detector.load_forecasting_model("1")
        called_path = mock_load.call_args[0][0]
        assert called_path.parent == models_dir_old


# ---------------------------------------------------------------------------
# build_anomaly_detector


class TestBuildAnomalyDetector:
    def test_uses_configured_scorer(self, sample_config):
        sample_config["detect"]["scorer_name"] = "NormScorer"
        sample_config["detect"]["scorer_params"] = {}
        detector = AnomalyDetector(sample_config)
        built = detector.build_anomaly_detector()
        assert built.scorer_name == "NormScorer"

    def test_drops_window_agg_param(self, sample_config):
        sample_config["detect"]["scorer_name"] = "KMeansScorer"
        sample_config["detect"]["scorer_params"] = {"n_clusters": 3, "window_agg": "should_be_dropped"}
        detector = AnomalyDetector(sample_config)
        built = detector.build_anomaly_detector()
        assert "window_agg" not in built.scorer_params

    def test_renames_k_to_n_clusters(self, sample_config):
        sample_config["detect"]["scorer_name"] = "KMeansScorer"
        sample_config["detect"]["scorer_params"] = {"k": 5}
        detector = AnomalyDetector(sample_config)
        built = detector.build_anomaly_detector()
        assert "k" not in built.scorer_params
        assert built.scorer_params.get("n_clusters") == 5

    def test_does_not_mutate_original_config(self, sample_config):
        original_params = {"n_clusters": 3, "window_agg": "drop_me"}
        sample_config["detect"]["scorer_params"] = original_params.copy()
        sample_config["detect"]["scorer_name"] = "KMeansScorer"
        AnomalyDetector(sample_config).build_anomaly_detector()
        assert sample_config["detect"]["scorer_params"] == original_params


# ---------------------------------------------------------------------------
# Test-window helpers


class TestCalculateTestWindowDefault:
    def test_returns_tail_when_n_exceeds_hist_window(self, sample_config):
        df = pd.DataFrame({"v": range(500)})
        start, end = AnomalyDetector(sample_config)._calculate_test_window_default(df, hist_window=100)
        assert (start, end) == (400, 500)

    def test_clamps_to_zero_when_hist_window_exceeds_n(self, sample_config):
        df = pd.DataFrame({"v": range(50)})
        start, end = AnomalyDetector(sample_config)._calculate_test_window_default(df, hist_window=100)
        assert (start, end) == (0, 50)


class TestAdjustWindowForInsufficientData:
    def test_raises_when_far_too_short(self, sample_config):
        df = pd.DataFrame({"v": range(50)})  # below MIN_TRAIN_SIZE * 2 = 200
        with pytest.raises(InsufficientDataException, match="too short"):
            AnomalyDetector(sample_config)._adjust_window_for_insufficient_data(
                df, test_start_idx=40, test_end_idx=50, hist_window=100
            )

    def test_raises_when_test_start_below_min_train_size(self, sample_config):
        df = pd.DataFrame({"v": range(250)})
        with pytest.raises(InsufficientDataException, match="Not enough data before test window"):
            AnomalyDetector(sample_config)._adjust_window_for_insufficient_data(
                df, test_start_idx=50, test_end_idx=250, hist_window=100
            )

    def test_returns_unchanged_when_window_is_already_valid(self, sample_config):
        df = pd.DataFrame({"v": range(500)})
        start, end = AnomalyDetector(sample_config)._adjust_window_for_insufficient_data(
            df, test_start_idx=300, test_end_idx=400, hist_window=100
        )
        assert (start, end) == (300, 400)

    def test_adjusts_when_test_start_meets_min_train(self, sample_config):
        df = pd.DataFrame({"v": range(250)})
        start, end = AnomalyDetector(sample_config)._adjust_window_for_insufficient_data(
            df, test_start_idx=150, test_end_idx=250, hist_window=100
        )
        # Already valid: start >= MIN_TRAIN_SIZE (100) and test window full.
        assert start == 150
        assert end == 250


# ---------------------------------------------------------------------------
# _split_unseen_scoring_data


class TestSplitUnseenScoringData:
    def _make_df(self, n=200):
        idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({"v": range(n)}, index=idx)

    def test_prefers_score_start_timestamp(self, sample_config):
        """``score_start_timestamp`` is the strictest leakage boundary —
        rows ``< cutoff`` are pipeline-seen, rows ``>= cutoff`` are the
        scorer's territory. Wins over legacy ``train_end_timestamp``."""
        df = self._make_df()
        score_start = df.index[180]
        # Include a stale (looser) train_end too to assert the order of preference.
        model_data = {"score_start_timestamp": score_start, "train_end_timestamp": df.index[120]}
        history, unseen = AnomalyDetector(sample_config)._split_unseen_scoring_data("1", df, model_data)
        assert len(history) == 180  # < cutoff
        assert len(unseen) == 20  # >= cutoff
        assert unseen.index.min() == score_start

    def test_uses_train_end_timestamp_when_available(self, sample_config):
        df = self._make_df()
        cutoff = df.index[120]
        model_data = {"train_end_timestamp": cutoff}
        history, unseen = AnomalyDetector(sample_config)._split_unseen_scoring_data("1", df, model_data)
        assert len(history) == 121  # inclusive of cutoff (<=)
        assert len(unseen) == 79
        assert unseen.index.min() > cutoff

    def test_falls_back_to_train_size_when_no_timestamp(self, sample_config):
        df = self._make_df()
        model_data = {"train_size": 120}
        history, unseen = AnomalyDetector(sample_config)._split_unseen_scoring_data("1", df, model_data)
        assert len(history) == 120
        assert len(unseen) == 80

    def test_falls_back_to_split_when_no_metadata(self, sample_config):
        df = self._make_df()
        sample_config["train"] = {"split": {"train": 60, "test": 10, "score": 30}}
        model_data: dict = {}
        history, unseen = AnomalyDetector(sample_config)._split_unseen_scoring_data("1", df, model_data)
        # train+test = 70% → first 140 rows are "seen", last 60 are unseen score window.
        assert len(history) == 140
        assert len(unseen) == 60

    def test_raises_when_no_unseen_data_left(self, sample_config):
        df = self._make_df()
        model_data = {"train_end_timestamp": df.index[-1]}
        with pytest.raises(InsufficientDataException, match="no unseen rows"):
            AnomalyDetector(sample_config)._split_unseen_scoring_data("1", df, model_data)

    def test_empty_input_returns_empty(self, sample_config):
        empty = pd.DataFrame(columns=["v"])
        history, unseen = AnomalyDetector(sample_config)._split_unseen_scoring_data(
            "1", empty, {"train_end_timestamp": pd.Timestamp("2025-01-01", tz="UTC")}
        )
        assert len(history) == 0
        assert len(unseen) == 0


# ---------------------------------------------------------------------------
# _validate_scoring_inputs


class TestValidateScoringInputs:
    def test_passes_on_clean_data(self, sample_config):
        df = pd.DataFrame({"v": np.arange(10.0)})
        AnomalyDetector(sample_config)._validate_scoring_inputs("1", df, df)

    def test_raises_on_nan_when_strict(self, sample_config):
        sample_config["detect"]["fail_on_nan_inputs"] = True
        bad = pd.DataFrame({"v": [1.0, np.nan, 3.0]})
        clean = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="invalid scoring inputs"):
            AnomalyDetector(sample_config)._validate_scoring_inputs("1", bad, clean)

    def test_raises_on_inf_when_strict(self, sample_config):
        sample_config["detect"]["fail_on_nan_inputs"] = True
        clean = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        bad = pd.DataFrame({"v": [1.0, np.inf, 3.0]})
        with pytest.raises(ValueError, match="invalid scoring inputs"):
            AnomalyDetector(sample_config)._validate_scoring_inputs("1", clean, bad)

    def test_warns_but_does_not_raise_when_not_strict(self, sample_config):
        sample_config["detect"]["fail_on_nan_inputs"] = False
        bad = pd.DataFrame({"v": [1.0, np.nan, 3.0]})
        clean = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        # Must not raise.
        AnomalyDetector(sample_config)._validate_scoring_inputs("1", bad, clean)


# ---------------------------------------------------------------------------
# _exclude_imputed_rows


class TestExcludeImputedRows:
    def test_drops_rows_with_low_weight(self, sample_config):
        idx = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
        source = pd.DataFrame(
            {
                "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "sensor_a__weight": [1.0, 1.0, 0.0, 1.0, 1.0],
            },
            index=idx,
        )
        aligned = source[["sensor_a"]]
        kept, mask = AnomalyDetector(sample_config)._exclude_imputed_rows(
            panel_id="1", window_name="test", source_df=source, aligned_true_df=aligned
        )
        assert len(kept) == 4
        assert mask.sum() == 4
        assert idx[2] not in kept.index

    def test_no_weight_columns_passes_through_unchanged(self, sample_config):
        idx = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
        source = pd.DataFrame({"sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        aligned = source[["sensor_a"]]
        kept, mask = AnomalyDetector(sample_config)._exclude_imputed_rows(
            panel_id="1", window_name="test", source_df=source, aligned_true_df=aligned
        )
        assert len(kept) == 5
        assert mask.all()

    def test_multiple_weight_cols_require_all_observed(self, sample_config):
        idx = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
        source = pd.DataFrame(
            {
                "sensor_a": [1.0, 2.0, 3.0, 4.0],
                "sensor_b": [10.0, 20.0, 30.0, 40.0],
                "sensor_a__weight": [1.0, 1.0, 1.0, 0.0],
                "sensor_b__weight": [1.0, 0.0, 1.0, 1.0],
            },
            index=idx,
        )
        aligned = source[["sensor_a", "sensor_b"]]
        kept, mask = AnomalyDetector(sample_config)._exclude_imputed_rows(
            panel_id="1", window_name="test", source_df=source, aligned_true_df=aligned
        )
        # Rows 1 and 3 are imputed in at least one feature -> dropped.
        assert len(kept) == 2
        assert list(kept.index) == [idx[0], idx[2]]
