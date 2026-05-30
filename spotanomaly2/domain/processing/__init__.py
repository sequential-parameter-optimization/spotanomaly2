"""Focused processing steps for the data pipeline (Strategy-style steps)."""

from spotanomaly2.domain.processing.base import ProcessingStep
from spotanomaly2.domain.processing.imputation import ImputationStep
from spotanomaly2.domain.processing.maintenance_removal import MaintenanceRemovalStep
from spotanomaly2.domain.processing.manual_outlier_removal import ManualOutlierRemovalStep
from spotanomaly2.domain.processing.median_filter import MedianFilterStep
from spotanomaly2.domain.processing.resample import ResampleStep
from spotanomaly2.domain.processing.temperature_aggregation import TemperatureAggregationStep
from spotanomaly2.domain.processing.weather_adjustment import WeatherAdjustmentStep

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
