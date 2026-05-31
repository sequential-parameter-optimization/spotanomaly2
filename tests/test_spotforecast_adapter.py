"""Tests for the spotforecast2 adapter (trainer + tuner)."""

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from spotanomaly2.domain.spotforecast_adapter import (
    _NAN_PENALTY,
    KernelRidgeApprox,
    SpotforecastPredictor,
    SpotforecastTrainer,
    SpotforecastTuner,
    SVRApprox,
    _apply_known_anomaly_imputation,
    _build_estimator,
    _build_nan_safe_metric,
    _create_forecaster,
)


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

    def test_train_panel_returns_eval_df(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        eval_df, timestamp = adapter.train_panel("test", panel_df)

        assert isinstance(eval_df, pd.DataFrame)
        assert "rmse" in eval_df.columns
        assert "mae" in eval_df.columns
        assert len(eval_df) == 2  # sensor_a, sensor_b
        assert isinstance(timestamp, str)

    def test_train_panel_completes_with_exclude_imputed_training_samples(self, sample_config, panel_df, tmp_path):
        """With ``exclude_imputed_training_samples=True`` the sample mask is a
        pd.Series, so the skip-sentinel check must not compare it element-wise
        (an old ``sample_mask == "skip"`` raised "truth value is ambiguous")."""
        config = copy.deepcopy(sample_config)
        config["paths"]["models_dir"] = str(tmp_path)
        config["train"]["lags"] = 6
        config["train"]["exclude_imputed_training_samples"] = True
        df = panel_df.copy()
        df["sensor_a__weight"] = 1.0
        df.loc[df.index[100:140], "sensor_a__weight"] = 0.0  # imputed window, plenty left

        eval_df, _ = SpotforecastTrainer(config).train_panel("test", df)

        assert isinstance(eval_df, pd.DataFrame)
        assert "sensor_a" in eval_df.index

    def test_predict_returns_dataframe(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        _, timestamp = adapter.train_panel("test", panel_df)

        model_path = Path(tmp_path) / timestamp / "fc_model_panel_test.pkl"
        model_data = joblib.load(model_path)

        test_slice = panel_df.iloc[-30:]
        history_slice = panel_df.iloc[:-30]
        predictor = SpotforecastPredictor(adapter.config)
        pred_df = predictor.predict(model_data, test_slice, history_df=history_slice)

        assert isinstance(pred_df, pd.DataFrame)
        assert len(pred_df) == 30
        assert set(pred_df.columns) == {"sensor_a", "sensor_b"}
        assert not pred_df.isna().all().any()

    def test_saved_model_format(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        _, timestamp = adapter.train_panel("test", panel_df)

        model_path = Path(tmp_path) / timestamp / "fc_model_panel_test.pkl"
        model_data = joblib.load(model_path)

        for key in ("forecasters", "n_lags", "target_cols", "exog_columns", "model_type"):
            assert key in model_data
        assert model_data["model_type"] == "spotforecast2_lightgbm"
        assert set(model_data["target_cols"]) == {"sensor_a", "sensor_b"}

    def test_train_eval_uses_one_step_ahead_for_long_horizons(self, sample_config, tmp_path):
        """Train-eval MAE for non-tree models must reflect one-step-ahead
        accuracy on real lags, not recursive divergence over the full test
        window. Recursive ``forecaster.predict(steps=N)`` on a kernel-style
        regressor cascades beyond the data range within a few hundred steps;
        one-step-ahead stays close to the noise floor.
        """
        rng = np.random.default_rng(0)
        n = 2000  # 1800 train / 200 test under default 0.9 ratio
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        seasonal = 50 * np.sin(np.arange(n) / 100)
        drift = np.linspace(0, 30, n)
        noise = rng.standard_normal(n) * 5
        y = 100 + seasonal + drift + noise
        df = pd.DataFrame({"sensor_a": y}, index=idx)

        channel_cfg = tmp_path / "panel_test.yaml"
        channel_cfg.write_text("default:\n  model: SVRApprox\n  params: {n_components: 256, C: 1.0}\n")

        config = copy.deepcopy(sample_config)
        config["paths"]["models_dir"] = str(tmp_path)
        config["train"]["channel_config_files"] = {"test": str(channel_cfg)}

        adapter = SpotforecastTrainer(config)
        eval_df, _ = adapter.train_panel("test", df)

        mae = float(eval_df.loc["sensor_a", "mae"])
        # One-step-ahead on this signal should land near the ~5 noise floor;
        # recursive predict over 200 steps would drift far beyond 30.
        assert mae < 15.0, f"Train eval MAE looks recursive (got {mae:.3f})"

    def test_per_channel_models_are_applied(self, adapter, panel_df, tmp_path):
        """Per-channel YAML overrides must reach the fitted estimators and the saved metadata."""
        from sklearn.linear_model import Ridge

        channel_cfg_path = tmp_path / "panel_test.yaml"
        channel_cfg_path.write_text(
            "default:\n"
            "  model: LightGBM\n"
            "channels:\n"
            "  sensor_a:\n"
            "    model: SVRApprox\n"
            "    params: {n_components: 64, C: 0.5}\n"
            "  sensor_b:\n"
            "    model: Ridge\n"
            "    params: {alpha: 1.0}\n"
        )
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        adapter.config["train"]["channel_config_files"] = {"test": str(channel_cfg_path)}

        _, timestamp = adapter.train_panel("test", panel_df)

        model_path = Path(tmp_path) / timestamp / "fc_model_panel_test.pkl"
        model_data = joblib.load(model_path)

        assert model_data["model_type"] == "spotforecast2_multi"
        assert model_data["model_name"] == "Multi"
        assert model_data["channel_models"]["sensor_a"]["model"] == "SVRApprox"
        assert model_data["channel_models"]["sensor_b"]["model"] == "Ridge"
        assert isinstance(model_data["forecasters"]["sensor_a"].estimator, SVRApprox)
        assert isinstance(model_data["forecasters"]["sensor_b"].estimator, Ridge)


SUPPORTED_MODELS = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "Ridge",
    "ElasticNet",
    "Lasso",
    "BayesianRidge",
    "Huber",
    "KernelRidgeApprox",
    "SVRApprox",
    "MLP",
]

NON_TREE_MODELS = [
    "Ridge",
    "ElasticNet",
    "Lasso",
    "BayesianRidge",
    "Huber",
    "KernelRidgeApprox",
    "SVRApprox",
    "MLP",
]


class TestBuildEstimator:
    """Direct tests for the model-name → sklearn estimator factory."""

    @pytest.mark.parametrize("name", SUPPORTED_MODELS)
    def test_build_returns_fittable_estimator(self, name):
        estimator = _build_estimator(name, {}, random_seed=42)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            _build_estimator("not_a_real_model", {}, random_seed=42)

    def test_unsupported_params_are_filtered(self):
        """Passing a kwarg that the base class does not support must not error."""
        estimator = _build_estimator(
            "ElasticNet",
            {"alpha": 0.5, "definitely_not_a_param": 123},
            random_seed=42,
        )
        assert estimator.alpha == 0.5


class TestCreateForecaster:
    """Verify the ForecasterRecursive helper plumbs scaling correctly."""

    @pytest.mark.parametrize("name", NON_TREE_MODELS)
    def test_non_tree_models_get_transformer_y(self, name):
        """Scale-sensitive models must receive a StandardScaler on the target."""
        forecaster = _create_forecaster(name, {}, n_lags=6, has_exog=False)
        assert forecaster.transformer_y is not None
        assert forecaster.transformer_exog is None

    @pytest.mark.parametrize("name", NON_TREE_MODELS)
    def test_non_tree_models_get_transformer_exog_when_exog_present(self, name):
        forecaster = _create_forecaster(name, {}, n_lags=6, has_exog=True)
        assert forecaster.transformer_y is not None
        assert forecaster.transformer_exog is not None

    @pytest.mark.parametrize("name", ["LightGBM", "XGBoost", "CatBoost"])
    def test_tree_models_skip_transformers(self, name):
        """Tree models don't need scaling — leaving transformers unset keeps fits cheap."""
        forecaster = _create_forecaster(name, {}, n_lags=6, has_exog=True)
        assert forecaster.transformer_y is None
        assert forecaster.transformer_exog is None

    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("ElasticNet", {"alpha": 0.01, "l1_ratio": 0.5}),
            ("Lasso", {"alpha": 0.01}),
            ("BayesianRidge", {}),
            ("Huber", {"alpha": 0.001, "epsilon": 1.5}),
        ],
    )
    def test_recursive_predict_stays_in_range_for_linear_models(self, name, kwargs):
        """Regression test for the LinearModel-bypass overflow.

        Without ``transformer_y``, ForecasterRecursive's fast path
        ``np.dot(X_raw, coef_) + intercept_`` cascades to ±inf within a few
        recursive steps when channel scales differ from the fitted-coef scale.
        With spotforecast2's native ``transformer_y`` / ``transformer_exog``
        in place, predictions stay finite and in the data's range.
        """
        rng = np.random.default_rng(0)
        n = 600
        y = pd.Series(
            100 + 20 * np.sin(np.arange(n) / 50) + rng.standard_normal(n) * 2,
            index=pd.date_range("2026-01-01", periods=n, freq="5min"),
            name="y",
        )
        forecaster = _create_forecaster(name, dict(kwargs), n_lags=24, has_exog=False)
        forecaster.fit(y=y.iloc[:500])
        pred = forecaster.predict(steps=20)

        assert np.all(np.isfinite(pred.values))
        # Values must stay in the same order of magnitude as the data.
        assert pred.values.min() > 50
        assert pred.values.max() < 200


class TestSpotforecastTuner:
    """Tune step must mirror train's feature topology + imputation handling."""

    @pytest.fixture
    def panel_df_with_exogenous(self):
        rng = np.random.default_rng(0)
        n = 400
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        return pd.DataFrame(
            {
                "sensor_a": rng.standard_normal(n) + 10,
                "sensor_b": rng.standard_normal(n) + 20,
                "exogenous_zulauf": rng.standard_normal(n) + 5,
            },
            index=idx,
        )

    def _tune_config(self):
        return {
            "n_trials": 3,
            "n_initial": 2,
            "metric": "mean_absolute_error",
            "models": ["LightGBM"],
            "model_search_spaces": {"LightGBM": {"num_leaves": [8, 31]}},
        }

    def test_tune_treats_exogenous_columns_as_exog_not_targets(self, sample_config, panel_df_with_exogenous):
        """Train fits with `exogenous_*` as features. Tune must do the same so
        the hyperparameters it picks are optimal for the model train will
        actually fit."""
        config = copy.deepcopy(sample_config)
        config["train"]["lags"] = 6
        tuner = SpotforecastTuner(config)
        results = tuner.tune_panel("test", panel_df_with_exogenous, self._tune_config())

        assert "sensor_a" in results
        assert "sensor_b" in results
        assert "exogenous_zulauf" not in results, "exogenous_* columns must be exog features, not tuning targets"

    def test_tune_uses_full_data_no_outer_holdout(self, sample_config):
        """The tuner's CV must:

        - Slice off the ``train.split.score`` percentage entirely (those rows
          belong to the scorer; the tuner must never see them).
        - Inside the train+test portion, split at the train/test boundary so
          CV-train mirrors what the trainer fits on and CV-val mirrors the
          trainer's test window.

        Sentinel check: with the standard 80/10/10 split, on N rows the CV
        ``initial_train_size`` must be ``int(N * 0.80)`` (NOT ``int((N*0.9) * 0.8)``).
        If a future change re-introduces a percentage-of-available split inside
        CV (e.g. 80% of the 90% pool), this test will catch it.
        """
        rng = np.random.default_rng(0)
        n = 400
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.9 * y[t - 1] + rng.standard_normal() * 0.1
        df = pd.DataFrame({"sensor_a": y + 10.0}, index=idx)

        config = copy.deepcopy(sample_config)
        config["train"]["lags"] = 6
        tuner = SpotforecastTuner(config)
        # Capture the OneStepAheadFold that the tuner actually builds.
        from spotforecast2.model_selection import OneStepAheadFold

        captured: dict = {}
        real_init = OneStepAheadFold.__init__

        def spy(self, *args, **kwargs):
            captured["initial_train_size"] = kwargs.get("initial_train_size", args[0] if args else None)
            return real_init(self, *args, **kwargs)

        OneStepAheadFold.__init__ = spy
        try:
            tuner.tune_panel(
                "test",
                df,
                {
                    "n_trials": 3,
                    "n_initial": 2,
                    "metric": "r2",
                    "models": ["Ridge"],
                    "model_search_spaces": {"Ridge": {"alpha": (0.001, 10.0, "log10")}},
                },
            )
        finally:
            OneStepAheadFold.__init__ = real_init

        # CV train size is ``int(N * split.train / 100)``, computed against the
        # FULL N rows, not against the (train + test) sub-pool. The score window
        # is sliced off before CV runs.
        split = config["train"]["split"]
        expected = int(n * split["train"] / 100)
        assert captured["initial_train_size"] == expected, (
            f"Tuner CV is splitting the wrong number of rows. Expected "
            f"initial_train_size={expected} ({split['train']}% of all {n} rows); "
            f"got {captured['initial_train_size']}."
        )

    def test_tune_uses_one_step_ahead_cv_on_autocorrelated_data(self, sample_config):
        """The tuner must score trials on one-step-ahead predictions (matching
        production), not multi-step recursive backtest. Recursive backtest
        cascade-diverges over thousands of steps for non-tree models and
        rewards trivial-mean predictors as "least bad".

        Regression check: on a strongly autocorrelated AR(1) series, lag-1
        achieves R² ≈ 0.99. The tuner must report an R² well above 0 — under
        the old recursive CV this would have been catastrophically negative.
        """
        rng = np.random.default_rng(0)
        # AR(1) with phi=0.95 → lag-1 explains ~90% of variance.
        n = 500
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.95 * y[t - 1] + rng.standard_normal() * 0.1
        df = pd.DataFrame({"sensor_a": y + 10.0}, index=idx)

        config = copy.deepcopy(sample_config)
        config["train"]["lags"] = 6
        # This test specifically targets the raw-target R² behaviour. Under
        # differentiation the target is Δy, whose variance for an AR(1)
        # process is dominated by noise → R²(Δy) ≈ 0.025 even with a
        # perfect lag-1 weight. That doesn't disprove one-step-ahead CV,
        # so disable differentiation here and let R²(raw y) be the signal.
        config["train"]["differentiation"] = 0
        tuner = SpotforecastTuner(config)
        tune_config = {
            "n_trials": 5,
            "n_initial": 3,
            "metric": "r2",
            "models": ["Ridge"],
            "model_search_spaces": {"Ridge": {"alpha": (0.001, 10.0, "log10")}},
        }
        results = tuner.tune_panel("test", df, tune_config)

        # negated R² → -1 = perfect, 0 = constant-mean baseline, +∞ = catastrophic.
        # With one-step-ahead CV on AR(1), Ridge should land well below -0.5.
        # With the old recursive CV this was wildly positive (R² ≪ 0).
        assert results["sensor_a"]["best_metric"] < -0.5, (
            f"Tuner is not using one-step-ahead CV — got metric "
            f"{results['sensor_a']['best_metric']:.3f}, expected < -0.5"
        )

    def test_tuner_r2_matches_production_r2_under_differentiation(self, sample_config):
        """Under differentiation, the tuner's reported R² must match what
        production / live mode reports for the *same* forecaster on the
        *same* val window. Same numerator (residuals), same denominator
        (Var(raw y)) — no more "tuner says 0.31, prod shows 0.91" gaps.

        Tested both with and without ``transformer_y`` (Ridge has it,
        LightGBM doesn't) — the metric must recover the scaler's σ
        exactly so scaled residuals integrate back to the right raw
        magnitude before R² is computed.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler
        from spotforecast2.model_selection import OneStepAheadFold
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.forecaster.utils import transform_numpy

        from spotanomaly2.domain.spotforecast_adapter import (
            _build_raw_r2_under_differentiation,
            _difference,
            _integrate_one_step,
        )

        rng = np.random.default_rng(0)
        n = 800
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.95 * y[t - 1] + rng.standard_normal() * 0.5
        y += 50.0
        s = pd.Series(y, index=idx, name="y")
        s_diff = _difference(s, 1).dropna()
        fit_size = int(len(s_diff) * 0.8)
        cv = OneStepAheadFold(initial_train_size=fit_size, verbose=False)

        def production_r2(needs_scaling: bool) -> float:
            """Mirror the predict() / detect() path exactly."""
            fc = ForecasterRecursive(
                estimator=Ridge(alpha=0.5),
                lags=6,
                transformer_y=StandardScaler() if needs_scaling else None,
            )
            fc.fit(y=s_diff.iloc[:fit_size])
            x, y_tgt = fc.create_train_X_y(y=s_diff)
            preds = fc.estimator.predict(x)
            if fc.transformer_y is not None:
                preds = transform_numpy(
                    np.asarray(preds, dtype=float),
                    fc.transformer_y,
                    fit=False,
                    inverse_transform=True,
                )
            preds_diff = pd.Series(preds, index=y_tgt.index)
            val_idx = s_diff.iloc[fit_size:].index
            abs_preds = _integrate_one_step(preds_diff.loc[val_idx], s, 1).dropna()
            return float(r2_score(s.reindex(abs_preds.index), abs_preds))

        def tuner_r2(needs_scaling: bool) -> float:
            """Drive the metric through ``spotoptim_search_forecaster`` —
            the actual code path the tuner uses. That path strips the
            index off ``y_true``/``y_train`` (passes ndarrays), so the
            metric MUST handle that case correctly. Using
            ``backtesting_forecaster`` here would mask a regression
            because it passes Series with intact ``.index``.
            """
            from spotforecast2.model_selection import spotoptim_search_forecaster

            fc = ForecasterRecursive(
                estimator=Ridge(),
                lags=6,
                transformer_y=StandardScaler() if needs_scaling else None,
            )
            new_metric = _build_raw_r2_under_differentiation(s)
            results, _ = spotoptim_search_forecaster(
                forecaster=fc,
                y=s_diff,
                cv=cv,
                search_space={"alpha": (0.4, 0.6, "log10")},
                metric=new_metric,
                return_best=True,
                random_state=42,
                verbose=False,
                n_trials=3,
                n_initial=2,
                show_progress=False,
            )
            return -float(results["r2_raw_after_integration"].iloc[0])

        # Ridge — scaled path (StandardScaler on Δy target). Match should
        # be within reporting precision (<0.001 R²): only the scaler's σ
        # recovery + spotforecast2's window_size trimming separate the two.
        prod_scaled = production_r2(needs_scaling=True)
        tune_scaled = tuner_r2(needs_scaling=True)
        assert abs(tune_scaled - prod_scaled) < 1e-3, (
            f"Tuner R² (scaled) drifted from production: tuner={tune_scaled:.6f}, prod={prod_scaled:.6f}"
        )

        # Ridge without transformer_y — same R² formula in either space,
        # so the only remaining gap is backtesting's internal window
        # accounting. Still well under 0.001 R².
        prod_raw = production_r2(needs_scaling=False)
        tune_raw = tuner_r2(needs_scaling=False)
        assert abs(tune_raw - prod_raw) < 1e-3, (
            f"Tuner R² (raw) drifted from production: tuner={tune_raw:.6f}, prod={prod_raw:.6f}"
        )

    def test_differentiation_closes_the_trivial_mean_trap(self, sample_config, tmp_path):
        """End-to-end: under differentiation, a degenerate hyperparameter
        config that previously collapsed to ``mean(y_train)`` at live time
        instead reduces to the lag-1 baseline at live time.

        We deliberately fit XGBoost with the *exact* parameter signature that
        kept winning the tuner historically — tiny ``learning_rate`` × few
        ``n_estimators`` × heavy reg — on a time-series that drifts away
        from the training mean. Without differentiation the prediction
        collapses to a near-constant ``base_score = mean(y_train)``, scoring
        wildly negative R² on live. With differentiation the same params
        produce ``y_pred[t] ≈ y[t-1]`` after one-step integration, scoring
        comfortably positive R² on live."""
        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        # AR(1) with a slow drift — the live period (last 10%) is offset
        # several σ below the training mean, exactly like cdp20.1.
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.95 * y[t - 1] + rng.standard_normal() * 0.5
        y += 50.0
        y[int(n * 0.9) :] -= 7.0
        df = pd.DataFrame({"sensor_a": y}, index=idx)

        # Force the degenerate XGBoost config that was winning before.
        degenerate_yaml = tmp_path / "panel_test.yaml"
        degenerate_yaml.write_text(
            "default:\n  model: XGBoost\n  params:\n"
            "    n_estimators: 100\n    learning_rate: 0.001\n"
            "    max_depth: 8\n    reg_alpha: 50.0\n    reg_lambda: 50.0\n"
        )

        def _train_and_score(diff_order: int) -> float:
            cfg = copy.deepcopy(sample_config)
            cfg["paths"]["models_dir"] = str(tmp_path / f"models_diff{diff_order}")
            cfg["train"]["channel_config_files"] = {"test": str(degenerate_yaml)}
            cfg["train"]["differentiation"] = diff_order
            cfg["train"]["lags"] = 6
            trainer = SpotforecastTrainer(cfg)
            _, ts = trainer.train_panel("test", df)
            model_data = joblib.load(Path(cfg["paths"]["models_dir"]) / ts / "fc_model_panel_test.pkl")
            n_local = len(df)
            test_slice = df.iloc[int(n_local * 0.9) :]
            history = df.iloc[: int(n_local * 0.9)]
            predictor = SpotforecastPredictor(cfg)
            preds = predictor.predict(model_data, test_slice, history_df=history)
            pred = preds["sensor_a"].dropna()
            truth = test_slice["sensor_a"].reindex(pred.index).dropna()
            pred = pred.reindex(truth.index)
            from sklearn.metrics import r2_score

            return float(r2_score(truth, pred))

        r2_raw = _train_and_score(diff_order=0)
        r2_diff = _train_and_score(diff_order=1)

        # Without differentiation, the degenerate model collapses to the
        # training mean and scores catastrophically negative on the shifted
        # live window.
        assert r2_raw < 0.0, (
            f"Sanity: the degenerate XGBoost config should fail without differentiation, but live R² = {r2_raw:+.3f}"
        )
        # With differentiation, the same params can't go below the lag-1
        # baseline. For AR(1)+drift, lag-1 R² is well above 0.
        assert r2_diff > 0.5, (
            f"Differentiation should pin the live floor near the lag-1 baseline (R² ≫ 0), but got {r2_diff:+.3f}"
        )

    def test_tune_handles_imputed_rows_when_exclusion_enabled(self, sample_config, panel_df_with_exogenous):
        """Row-level imputed-sample exclusion is a TRAIN-ONLY feature (audit C2):
        SpotforecastTrainer applies it via forecaster.weight_func, but the
        one-step-ahead tuning objective fits the bare estimator and ignores
        weight_func, so tuning cannot drop individual imputed rows. This test pins
        the honest contract: with `exclude_imputed_training_samples=True` and
        imputed rows present, the search still completes and yields a finite metric
        (imputed rows are interpolated for lag context, and the observed mask only
        gates channel skipping)."""
        df = panel_df_with_exogenous.copy()
        df["sensor_a__weight"] = 1.0
        # Mark a window of rows as imputed.
        df.loc[df.index[100:200], "sensor_a__weight"] = 0.0

        config = copy.deepcopy(sample_config)
        config["train"]["lags"] = 6
        config["train"]["exclude_imputed_training_samples"] = True

        tuner = SpotforecastTuner(config)
        results = tuner.tune_panel("test", df, self._tune_config())

        assert "sensor_a" in results
        assert "error" not in results["sensor_a"]
        assert results["sensor_a"]["best_metric"] is not None
        assert np.isfinite(results["sensor_a"]["best_metric"])


class TestNystroemApprox:
    """Nyström-backed variants are the only kernel models in the project;
    they replace the exact RBF SVR / KernelRidge that needed a row cap."""

    def test_exact_kernel_models_are_no_longer_supported(self):
        """Exact RBF SVR / KernelRidge were removed because their N×N Gram
        matrix forces a row cap that biases tuning and degrades production
        predictions. ``_build_estimator`` must reject them."""
        with pytest.raises(ValueError, match="Unsupported model"):
            _build_estimator("SVR", {}, random_seed=0)
        with pytest.raises(ValueError, match="Unsupported model"):
            _build_estimator("KernelRidge", {}, random_seed=0)

    def test_kernel_ridge_approx_fits_and_predicts(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.standard_normal((n, 4))
        y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.standard_normal(n) * 0.05

        est = KernelRidgeApprox(n_components=64, gamma=0.5, alpha=0.1, random_state=0)
        est.fit(x, y)
        preds = est.predict(x)
        assert preds.shape == (n,)
        # MAE should be small on a smooth signal — Nyström approximates the
        # exact KernelRidge well at this n_components.
        assert float(np.mean(np.abs(preds - y))) < 0.5

    def test_svr_approx_fits_and_predicts(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.standard_normal((n, 4))
        y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.standard_normal(n) * 0.05

        est = SVRApprox(n_components=64, gamma=0.5, C=1.0, epsilon=0.05, random_state=0)
        est.fit(x, y)
        preds = est.predict(x)
        assert preds.shape == (n,)
        assert float(np.mean(np.abs(preds - y))) < 0.5

    def test_kernel_ridge_approx_honors_sample_weight(self):
        """Ridge accepts sample_weight directly — zero-weight rows must not
        steer the fit."""
        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal((n, 2))
        y = x[:, 0] + 0.1 * rng.standard_normal(n)
        # Inject huge outliers in the second half, then mask them out.
        y[150:] += 1000.0
        sample_weight = np.ones(n)
        sample_weight[150:] = 0.0

        est = KernelRidgeApprox(n_components=64, gamma=0.5, alpha=0.1, random_state=0)
        est.fit(x, y, sample_weight=sample_weight)

        # Predictions on the (clean) first half should ignore the masked outliers.
        preds = est.predict(x[:150])
        assert float(np.mean(np.abs(preds - y[:150]))) < 0.5

    def test_mlp_handles_zero_weight_rows(self):
        """``MLPRegressor`` calls ``np.average(..., weights=sample_weight)``
        inside its loss and a batch (or the early-stopping val slice) made
        entirely of zero-weight rows raises *Weights sum to zero, can't be
        normalized*. Our wrapper drops zero-weight rows up front so this
        cannot happen, and the fit reflects only the kept rows."""
        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal((n, 4))
        y = x[:, 0] + 0.1 * rng.standard_normal(n)
        # Add huge outliers and then mask them out via sample_weight.
        y[150:] += 1000.0
        sample_weight = np.ones(n)
        sample_weight[150:] = 0.0

        est = _build_estimator("MLP", {"hidden_layer_sizes": (16,), "max_iter": 200}, random_seed=0)
        est.fit(x, y, sample_weight=sample_weight)

        # The wrapper must have dropped the masked rows — predictions on the
        # clean half should track the underlying y ≈ x[:, 0], not the +1000
        # outliers. (~1.0 MAE leaves headroom for MLP noise; without the drop
        # the fit either crashes or chases the outliers.)
        preds = est.predict(x[:50])
        assert float(np.mean(np.abs(preds - y[:50]))) < 1.0

    def test_svr_approx_drops_zero_weight_rows(self):
        """LinearSVR doesn't accept sample_weight; the wrapper must drop
        zero-weight rows before fit so masking still works."""
        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal((n, 2))
        y = x[:, 0] + 0.1 * rng.standard_normal(n)
        y[150:] += 1000.0
        sample_weight = np.ones(n)
        sample_weight[150:] = 0.0

        est = SVRApprox(n_components=64, gamma=0.5, C=1.0, epsilon=0.05, random_state=0)
        est.fit(x, y, sample_weight=sample_weight)

        preds = est.predict(x[:150])
        assert float(np.mean(np.abs(preds - y[:150]))) < 1.0


class TestNanSafeMetric:
    """The metric wrapper feeds SpotOptim a single scalar to *minimise*. R²
    must therefore be negated, and any NaN/Inf trial must collapse to the
    finite penalty so the GP surrogate doesn't crash."""

    def test_mae_passes_through_lower_is_better(self):
        metric = _build_nan_safe_metric("mae")
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.2])
        # Plain MAE: positive, minimised when small.
        assert metric(y_true, y_pred) == pytest.approx(0.1, abs=1e-9)

    def test_r2_is_negated_so_minimiser_finds_better_fits(self):
        metric = _build_nan_safe_metric("r2")
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perfect fit: r2_score = 1.0 → wrapper returns -1.0.
        assert metric(y_true, y_true) == pytest.approx(-1.0, abs=1e-9)
        # Constant-mean predictor: r2_score = 0 → wrapper returns 0.
        # The trivial-mean trap that MAE/MSE fall into now sits *exactly*
        # on the SpotOptim baseline; any real model with R² > 0 wins.
        mean_pred = np.full_like(y_true, y_true.mean())
        assert metric(y_true, mean_pred) == pytest.approx(0.0, abs=1e-9)

    def test_r2_orders_real_fit_better_than_constant_predictor(self):
        """The whole point of the switch: a model that tracks variation must
        beat a constant predictor under the wrapped metric, even on
        narrow-range targets where MAE would tie them."""
        metric = _build_nan_safe_metric("r2")
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        good_pred = y_true + 0.05  # slight bias, tracks variation
        constant_pred = np.full_like(y_true, y_true.mean())
        assert metric(y_true, good_pred) < metric(y_true, constant_pred)

    def test_nan_predictions_become_finite_penalty(self):
        """Divergent MLP / Bayesian trials must not poison the GP fit."""
        metric_r2 = _build_nan_safe_metric("r2")
        metric_mae = _build_nan_safe_metric("mae")
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred_nan = np.full_like(y_true, np.nan)
        assert metric_r2(y_true, y_pred_nan) == _NAN_PENALTY
        assert metric_mae(y_true, y_pred_nan) == _NAN_PENALTY

    def test_unknown_metric_falls_through(self):
        """Strings the wrapper doesn't recognise are returned verbatim so
        spotforecast2 can resolve them itself."""
        out = _build_nan_safe_metric("custom_unknown_metric")
        assert out == "custom_unknown_metric"


class TestSpotforecastPredictor:
    """Inference against a trained model artifact, separate from training."""

    def test_predict_returns_dataframe_indexed_to_input(self, adapter, panel_df, tmp_path):
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        _, timestamp = adapter.train_panel("test", panel_df)
        model_data = joblib.load(Path(tmp_path) / timestamp / "fc_model_panel_test.pkl")

        test_slice = panel_df.iloc[-30:]
        history_slice = panel_df.iloc[:-30]
        predictor = SpotforecastPredictor(adapter.config)
        pred_df = predictor.predict(model_data, test_slice, history_df=history_slice)

        assert list(pred_df.index) == list(test_slice.index)
        assert set(pred_df.columns) == set(model_data["target_cols"])

    def test_predict_without_history_still_returns_finite_predictions(self, adapter, panel_df, tmp_path):
        """``history_df=None`` is the supported live-mode signature; the
        predictor should fall back to using ``df`` as its own history."""
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        _, timestamp = adapter.train_panel("test", panel_df)
        model_data = joblib.load(Path(tmp_path) / timestamp / "fc_model_panel_test.pkl")

        predictor = SpotforecastPredictor(adapter.config)
        pred_df = predictor.predict(model_data, panel_df.iloc[-50:])
        # At least some predictions should resolve - the recursive lag block at
        # the front is NaN, but the bulk of the window should be finite.
        assert pred_df.notna().any().all()

    def test_missing_forecaster_yields_nan_column(self, adapter, panel_df, tmp_path):
        """If a target column's forecaster is missing from the artifact
        (e.g. trainer skipped it), the predictor returns an all-NaN column
        instead of raising — keeps detection alive on partial models."""
        adapter.config["paths"]["models_dir"] = str(tmp_path)
        _, timestamp = adapter.train_panel("test", panel_df)
        model_data = joblib.load(Path(tmp_path) / timestamp / "fc_model_panel_test.pkl")
        # Drop one channel's forecaster but keep the column in target_cols
        # so the predictor still iterates over it.
        del model_data["forecasters"]["sensor_a"]

        predictor = SpotforecastPredictor(adapter.config)
        pred_df = predictor.predict(model_data, panel_df.iloc[-30:], history_df=panel_df.iloc[:-30])

        assert pred_df["sensor_a"].isna().all()
        assert pred_df["sensor_b"].notna().any()


class TestTrainerHelpers:
    """Pure, side-effect-free helpers exposed for unit testing."""

    def test_resolve_channel_lags_passes_through_int(self):
        eff, n = SpotforecastTrainer._resolve_channel_lags({}, 24)
        assert eff == 24
        assert n == 24

    def test_resolve_channel_lags_prefers_best_lags_list_over_default(self):
        eff, n = SpotforecastTrainer._resolve_channel_lags({"best_lags": [1, 3, 7, 24]}, 12)
        assert eff == [1, 3, 7, 24]
        assert n == 24  # max of the list — used for sufficiency checks

    def test_resolve_channel_lags_best_lags_scalar_overrides_default(self):
        eff, n = SpotforecastTrainer._resolve_channel_lags({"best_lags": 6}, 24)
        assert eff == 6
        assert n == 6

    def test_resolve_channel_lags_invalid_best_lags_falls_back_to_default(self):
        eff, n = SpotforecastTrainer._resolve_channel_lags({"best_lags": "not-a-number"}, 24)
        assert eff == 24
        assert n == 24

    def test_resolve_channel_model_spec_channel_overrides_panel_default(self):
        name, params = SpotforecastTrainer._resolve_channel_model_spec(
            channel_cfg={"model": "Ridge", "params": {"alpha": 0.5}},
            panel_default_model="LightGBM",
            panel_default_params={"num_leaves": 31},
            model_label="LightGBM",
        )
        assert name == "Ridge"
        # Panel defaults must NOT bleed into the channel when the channel
        # picks a different model — those defaults belong to LightGBM.
        assert params == {"alpha": 0.5}

    def test_resolve_channel_model_spec_inherits_panel_default_params_when_models_match(self):
        name, params = SpotforecastTrainer._resolve_channel_model_spec(
            channel_cfg={"params": {"num_leaves": 63}},  # override one knob
            panel_default_model="LightGBM",
            panel_default_params={"num_leaves": 31, "learning_rate": 0.05},
            model_label="Ridge",
        )
        assert name == "LightGBM"
        # Channel params override panel defaults key-by-key, others inherited.
        assert params == {"num_leaves": 63, "learning_rate": 0.05}

    def test_resolve_channel_model_spec_falls_back_to_fallback_model(self):
        name, params = SpotforecastTrainer._resolve_channel_model_spec(
            channel_cfg={},
            panel_default_model=None,
            panel_default_params={},
            model_label="LightGBM",
        )
        assert name == "LightGBM"
        assert params == {}


def test_apply_known_anomaly_imputation_falls_back_to_linear_for_psm():
    # Regression: 'psm' is now a registered strategy, but it must NOT be used to
    # re-impute the small interior cells produced by known-anomaly masking (PSM
    # needs a freq-bearing DatetimeIndex and would otherwise raise). The helper
    # must transparently fall back to linear interpolation.
    # Build a DatetimeIndex with NO freq set: PSM would raise on it, linear won't.
    idx = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC").values, tz="UTC")
    assert idx.freq is None  # the condition that makes PSM raise but linear cope
    df = pd.DataFrame({"channel_1_ph": np.arange(10, dtype=float)}, index=idx)
    known_anomalies = [{"start": "2025-01-01 00:20", "end": "2025-01-01 00:25"}]

    out = _apply_known_anomaly_imputation(
        df,
        known_anomalies=known_anomalies,
        buffer="0min",
        target_cols=["channel_1_ph"],
        weight_suffix="__weight",
        imputation_method="psm",
    )

    # The masked cells were re-imputed (no NaNs, no PSM crash) ...
    assert out["channel_1_ph"].isna().sum() == 0
    # ... and flagged as not-real in the weight column.
    assert (out["channel_1_ph__weight"] == 0).any()
