"""Verify all modules import without error."""

import importlib

import pytest

MODULES = [
    "spotanomaly2.main",
    "spotanomaly2.application.config",
    "spotanomaly2.application.data_manager",
    "spotanomaly2.application.pipeline",
    "spotanomaly2.domain.anomaly_detector",
    "spotanomaly2.domain.constants",
    "spotanomaly2.domain.data_processor",
    "spotanomaly2.domain.exceptions",
    "spotanomaly2.domain.model_trainer",
    "spotanomaly2.domain.processing_steps",
    "spotanomaly2.domain.spotforecast_adapter",
    "spotanomaly2.infrastructure.logging",
    "spotanomaly2.infrastructure.storage",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_spotanomaly2_safe_reexport():
    from spotanomaly2_safe.scoring.pipeline import ForecastingAnomalyDetector

    assert ForecastingAnomalyDetector is not None
