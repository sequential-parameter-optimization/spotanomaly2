"""Adapter wrapping spotforecast2's ForecasterRecursive for spotanomaly2's pipeline.

spotanomaly2's own data-processing pipeline (5-minute resampling, imputation, ...)
runs *before* this adapter, so we go straight to ``ForecasterRecursive`` instead of
through ``MultiTask.prepare_data()`` and add only what's domain-specific:

- per-channel model selection from YAML
- training-sample weighting that respects spotanomaly2's imputation flags
- a cheap Ridge pre-pass to mask anomalous training regions
- batch one-step-ahead prediction over an arbitrary held-out frame

Feature scaling for non-tree models is delegated to spotforecast2's built-in
``transformer_y`` / ``transformer_exog`` rather than wrapping the estimator.

This module used to be a single 1.4 kLOC file; the implementation now lives in
sub-modules. The names below are re-exported here so existing imports of the
form ``from spotanomaly2.domain.spotforecast_adapter import …`` keep working.
"""

from .estimators import (
    _NEEDS_SCALING,
    KernelRidgeApprox,
    SVRApprox,
    _CatBoostRegressor,
    _MLPRegressorRobust,
)
from .factory import (
    _build_estimator,
    _create_forecaster,
    _filter_params,
)
from .prediction import (
    _difference,
    _integrate_one_step,
    _one_step_ahead_predict,
    _predict_one_step_integrated,
)
from .predictor import SpotforecastPredictor
from .preprocessing import (
    _apply_known_anomaly_imputation,
    _build_strict_training_sample_mask,
    _compute_observed_mask,
    _detect_anomalies_via_ridge,
    _ensure_freq,
    _interpolate_inplace,
    _mask_known_anomalies,
    _split_panel_columns,
)
from .trainer import SpotforecastTrainer
from .tuner import SpotforecastTuner
from .tuning_metrics import (
    _NAN_PENALTY,
    _build_nan_safe_metric,
    _build_raw_r2_under_differentiation,
    _yaml_search_space_to_dict,
)

__all__ = [
    "KernelRidgeApprox",
    "SVRApprox",
    "SpotforecastPredictor",
    "SpotforecastTrainer",
    "SpotforecastTuner",
    "_NAN_PENALTY",
    "_NEEDS_SCALING",
    "_apply_known_anomaly_imputation",
    "_build_estimator",
    "_build_nan_safe_metric",
    "_build_raw_r2_under_differentiation",
    "_build_strict_training_sample_mask",
    "_CatBoostRegressor",
    "_compute_observed_mask",
    "_create_forecaster",
    "_detect_anomalies_via_ridge",
    "_difference",
    "_ensure_freq",
    "_filter_params",
    "_integrate_one_step",
    "_interpolate_inplace",
    "_mask_known_anomalies",
    "_MLPRegressorRobust",
    "_one_step_ahead_predict",
    "_predict_one_step_integrated",
    "_split_panel_columns",
    "_yaml_search_space_to_dict",
]
