"""Focused processing steps for the data pipeline (Strategy-style steps)."""

from spotanomaly2.domain.processing_steps.base import ProcessingStep
from spotanomaly2.domain.processing_steps.imputation import ImputationStep
from spotanomaly2.domain.processing_steps.maintenance_removal import MaintenanceRemovalStep
from spotanomaly2.domain.processing_steps.manual_outlier_removal import ManualOutlierRemovalStep
from spotanomaly2.domain.processing_steps.median_filter import MedianFilterStep
from spotanomaly2.domain.processing_steps.resample import ResampleStep
from spotanomaly2.domain.processing_steps.temperature_aggregation import TemperatureAggregationStep
from spotanomaly2.domain.processing_steps.weather_adjustment import WeatherAdjustmentStep

__all__ = [
    "ImputationStep",
    "MaintenanceRemovalStep",
    "ManualOutlierRemovalStep",
    "MedianFilterStep",
    "ProcessingStep",
    "ResampleStep",
    "TemperatureAggregationStep",
    "WeatherAdjustmentStep",
]
