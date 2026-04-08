"""Test the SpotforecastTrainer adapter using the MultiTask-based API."""

import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.spotforecast_adapter import SpotforecastTrainer


@pytest.fixture
def adapter(sample_config):
    return SpotforecastTrainer(sample_config)


@pytest.fixture
def panel_df():
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "sensor_a": rng.standard_normal(n) + 10,
            "sensor_b": rng.standard_normal(n) + 20,
        },
        index=idx,
    )


class TestSpotforecastTrainer:
    def test_instantiation(self, adapter):
        assert adapter is not None
        assert adapter.config is not None

    def test_create_multitask(self, adapter, tmp_path):
        mt = adapter._create_multitask("test", n_lags=6, cache_home=tmp_path)
        assert mt is not None
        assert mt.TASK == "lazy"
        assert mt.config.use_exogenous_features is False

    def test_create_multitask_forecaster(self, adapter, tmp_path):
        mt = adapter._create_multitask("test", n_lags=6, cache_home=tmp_path)
        forecaster = mt.create_forecaster()
        assert forecaster is not None
        assert hasattr(forecaster, "fit")
        assert hasattr(forecaster, "predict")

    def test_train_panel_returns_eval_df(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        eval_df, timestamp = adapter.train_panel("test", panel_df)

        assert isinstance(eval_df, pd.DataFrame)
        assert "rmse" in eval_df.columns
        assert "mae" in eval_df.columns
        assert len(eval_df) == 2  # sensor_a, sensor_b
        assert isinstance(timestamp, str)

    def test_predict_returns_dataframe(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        eval_df, timestamp = adapter.train_panel("test", panel_df)

        from pathlib import Path

        import joblib

        model_path = Path(tmp_path) / timestamp / "LightGBM_fc_model_panel_test.pkl"
        model_data = joblib.load(model_path)

        test_slice = panel_df.iloc[-30:]
        history_slice = panel_df.iloc[:-30]

        pred_df = adapter.predict(model_data, test_slice, history_df=history_slice)

        assert isinstance(pred_df, pd.DataFrame)
        assert len(pred_df) == 30
        assert set(pred_df.columns) == {"sensor_a", "sensor_b"}
        assert not pred_df.isna().all().any()

    def test_saved_model_format(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        _, timestamp = adapter.train_panel("test", panel_df)

        from pathlib import Path

        import joblib

        model_path = Path(tmp_path) / timestamp / "LightGBM_fc_model_panel_test.pkl"
        model_data = joblib.load(model_path)

        assert "forecasters" in model_data
        assert "n_lags" in model_data
        assert "target_cols" in model_data
        assert "exog_columns" in model_data
        assert "model_type" in model_data
        assert model_data["model_type"] == "spotforecast2_lightgbm"
        assert set(model_data["target_cols"]) == {"sensor_a", "sensor_b"}
