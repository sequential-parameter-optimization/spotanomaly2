"""Test Pipeline instantiation and individual step mocking."""

from unittest.mock import MagicMock, patch

import pandas as pd

from spotanomaly2.application.pipeline import Pipeline


def test_pipeline_instantiation(sample_config):
    pipeline = Pipeline(sample_config)
    assert pipeline is not None
    assert pipeline.config == sample_config


def test_pipeline_detect_instantiates_detector(sample_config):
    """Verify detect() creates an AnomalyDetector and calls run()."""
    pipeline = Pipeline(sample_config)

    with (
        patch("spotanomaly2.application.pipeline.AnomalyDetector") as mock_detector,
        patch.object(pipeline._data_manager, "load_processed_data") as mock_load,
        patch.object(pipeline._data_manager, "save_detection_results"),
    ):
        mock_load.return_value = {"1": pd.DataFrame({"a": [1, 2, 3]})}
        mock_instance = MagicMock()
        mock_instance.run.return_value = {}
        mock_detector.return_value = mock_instance

        pipeline.detect()

        mock_detector.assert_called_once_with(sample_config, pipeline.logger)
        mock_instance.run.assert_called_once()


def test_pipeline_train_loads_processed_data(sample_config):
    """Verify train() loads processed data and iterates panels."""
    pipeline = Pipeline(sample_config)

    with patch.object(pipeline._data_manager, "load_processed_data") as mock_load:
        mock_load.return_value = {}

        pipeline.train()

        mock_load.assert_called_once()
