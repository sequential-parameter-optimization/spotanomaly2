# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the ModelTuner domain wrapper.

ModelTuner is the thin layer between Pipeline.tune and the SpotforecastTuner
adapter. It owns two pieces of pure logic that nothing else exercised:

  * ``_build_tune_config`` — merges the ``tune`` config section into the search
    spec the adapter consumes (candidate models, shared lags, panel overrides).
  * ``update_channel_configs`` — writes tuning winners back into the per-panel
    ``channel_models`` YAMLs that ``train`` later reads. This is the writer half
    of the tune->train seam; only the reader (load_panel_channel_config) was
    tested before.

These are deterministic (no real optimisation), so they stay fast.
"""

import numpy as np
import yaml

from spotanomaly2.application.config import load_panel_channel_config
from spotanomaly2.domain.model_tuner import ModelTuner

# ---------------------------------------------------------------------------
# _build_tune_config
# ---------------------------------------------------------------------------


class TestBuildTuneConfig:
    def test_defaults_to_lightgbm_when_no_tune_section(self, sample_config):
        tuner = ModelTuner(sample_config)
        cfg = tuner._build_tune_config()
        assert cfg["models"] == ["LightGBM"]
        assert cfg["model_search_spaces"] == {"LightGBM": {}}
        assert cfg["n_trials"] == 10
        assert cfg["n_initial"] == 5
        assert cfg["metric"] == "mean_absolute_error"

    def test_reads_trial_budget_and_metric(self, sample_config):
        sample_config["tune"] = {
            "n_trials": 25,
            "n_initial": 7,
            "metric": "r2",
            "models": {"LightGBM": {"num_leaves": [10, 50]}},
        }
        cfg = ModelTuner(sample_config)._build_tune_config()
        assert cfg["n_trials"] == 25
        assert cfg["n_initial"] == 7
        assert cfg["metric"] == "r2"
        assert cfg["models"] == ["LightGBM"]
        assert cfg["model_search_spaces"]["LightGBM"]["num_leaves"] == [10, 50]

    def test_shared_lags_injected_into_each_model_space(self, sample_config):
        sample_config["tune"] = {
            "lags": [1, 2, 3],
            "models": {"LightGBM": {}, "Ridge": {"alpha": [0.1, 1.0]}},
        }
        cfg = ModelTuner(sample_config)._build_tune_config()
        assert cfg["model_search_spaces"]["LightGBM"]["lags"] == [1, 2, 3]
        assert cfg["model_search_spaces"]["Ridge"]["lags"] == [1, 2, 3]
        # Existing per-model entries are preserved alongside the shared lags.
        assert cfg["model_search_spaces"]["Ridge"]["alpha"] == [0.1, 1.0]

    def test_explicit_model_lags_not_overwritten_by_shared(self, sample_config):
        sample_config["tune"] = {
            "lags": [1, 2, 3],
            "models": {"LightGBM": {"lags": [10, 20]}},
        }
        cfg = ModelTuner(sample_config)._build_tune_config()
        # setdefault must not clobber an explicit per-model lag spec.
        assert cfg["model_search_spaces"]["LightGBM"]["lags"] == [10, 20]

    def test_panel_overrides_passed_through(self, sample_config):
        sample_config["tune"] = {"panel_overrides": {"1": {"default": {"n_trials": 3}}}}
        cfg = ModelTuner(sample_config)._build_tune_config()
        assert cfg["panel_overrides"] == {"1": {"default": {"n_trials": 3}}}

    def test_invalid_models_mapping_raises(self, sample_config):
        sample_config["tune"] = {"models": ["LightGBM"]}  # list, not mapping
        tuner = ModelTuner(sample_config)
        try:
            tuner._build_tune_config()
        except ValueError as exc:
            assert "tune.models" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-mapping tune.models")


# ---------------------------------------------------------------------------
# update_channel_configs  (the tune -> train writeback seam)
# ---------------------------------------------------------------------------


class TestUpdateChannelConfigs:
    def _config_with_channel_file(self, sample_config, tmp_path):
        cfg_path = tmp_path / "panel_1.yaml"
        sample_config["train"]["channel_config_files"] = {"1": str(cfg_path)}
        return sample_config, cfg_path

    def test_writes_winner_and_coerces_numpy(self, sample_config, tmp_path):
        config, cfg_path = self._config_with_channel_file(sample_config, tmp_path)
        results = {
            "1": {
                "sensor_a": {
                    "best_model": "LightGBM",
                    "best_lags": np.array([1, 2, 3]),
                    "best_params": {"num_leaves": np.int64(31), "learning_rate": np.float64(0.05)},
                }
            }
        }
        ModelTuner(config).update_channel_configs(results)

        assert cfg_path.exists()
        written = yaml.safe_load(cfg_path.read_text())
        ch = written["channels"]["sensor_a"]
        assert ch["model"] == "LightGBM"
        # numpy types must be coerced to native Python so the YAML round-trips
        # and ``train`` reads plain ints/floats/lists.
        assert ch["best_lags"] == [1, 2, 3]
        assert isinstance(ch["params"]["num_leaves"], int)
        assert isinstance(ch["params"]["learning_rate"], float)
        assert ch["params"]["num_leaves"] == 31

    def test_skips_channels_that_errored(self, sample_config, tmp_path):
        config, cfg_path = self._config_with_channel_file(sample_config, tmp_path)
        results = {
            "1": {
                "sensor_a": {"best_model": "LightGBM", "best_lags": 6, "best_params": {}},
                "sensor_b": {"error": "all models failed"},
            }
        }
        ModelTuner(config).update_channel_configs(results)

        written = yaml.safe_load(cfg_path.read_text())
        assert "sensor_a" in written["channels"]
        assert "sensor_b" not in written["channels"]

    def test_panel_prefixed_key_fallback(self, sample_config, tmp_path):
        cfg_path = tmp_path / "panel_1.yaml"
        # Mapping keyed by ``panel_1`` instead of ``1`` must still resolve.
        sample_config["train"]["channel_config_files"] = {"panel_1": str(cfg_path)}
        results = {"1": {"sensor_a": {"best_model": "Ridge", "best_lags": 4, "best_params": {}}}}
        ModelTuner(sample_config).update_channel_configs(results)
        assert cfg_path.exists()
        assert yaml.safe_load(cfg_path.read_text())["channels"]["sensor_a"]["model"] == "Ridge"

    def test_written_config_is_readable_by_loader(self, sample_config, tmp_path):
        """The writeback must be consumable by the reader train uses."""
        config, cfg_path = self._config_with_channel_file(sample_config, tmp_path)
        results = {"1": {"sensor_a": {"best_model": "LightGBM", "best_lags": [2, 4], "best_params": {"n": 1}}}}
        ModelTuner(config).update_channel_configs(results)

        loaded = load_panel_channel_config("1", config)
        assert loaded["channels"]["sensor_a"]["model"] == "LightGBM"
        assert loaded["channels"]["sensor_a"]["best_lags"] == [2, 4]
