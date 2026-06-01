"""Shared fixtures for spotanomaly2 tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config():
    """Minimal config dict matching default.yaml structure."""
    return {
        "panels": {
            "panel_ids": ["1"],
            "channel_ids": ["0"],
        },
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "models_dir": "data/models",
            "results_dir": "data/results",
            "evaluations_dir": "data/evaluations",
            "credentials_file": ".env",
            "raw_data_version": None,
        },
        "process": {
            "resample": {"freq": "5min", "origin": "start_day", "label": "right", "closed": "right"},
            "maintenance_column": "channel_0_maintenance_flag",
            "imputation": {
                "method": "linear_interpolation",
                "params": {},
                "add_weights": True,
                "weight_suffix": "__weight",
            },
            "manual_outliers": {"enabled": False},
            "flow_columns_pattern": "flow",
            "temperature_columns_pattern": "temperature",
            "median_filter_kernel": 3,
        },
        "train": {
            "split": {"train": 80, "val": 10, "test": 10},
            "random_seed": 42,
            "fallback_model": "LightGBM",
            "lags": 6,
            "exog_columns": [],
        },
        "detect": {
            "hist_window": 100,
            "target_date": None,
            "high_quantile": 0.99,
            "scorer_name": "KMeansScorer",
            "scorer_params": {"n_clusters": 3, "window": 1},
            "normalize_scores": True,
            "normalization_quantile": 0.99,
        },
        "known_anomalies": [],
        "fetch": {},
        "primary": {"display_name": "Primary"},
        # New plugin-shape: empty list means no exogenous sources run.
        # Tests that need a source instantiate one explicitly. Per-source
        # ``multiply_residuals`` replaces the old global residual_weighting flag.
        "exogenous": [],
        "report": {"enabled": False},
    }


@pytest.fixture
def synthetic_panel_df():
    """Synthetic panel DataFrame with DatetimeIndex at 5min intervals."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "sensor_a": rng.standard_normal(n) + 10,
            "sensor_b": rng.standard_normal(n) + 20,
            "sensor_c": rng.standard_normal(n) + 30,
        },
        index=idx,
    )
