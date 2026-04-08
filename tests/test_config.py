"""Test configuration loading."""

from pathlib import Path

import pytest

from spotanomaly2.application.config import get_default_config_path, load_config, load_default_config


def test_default_config_path_exists():
    path = get_default_config_path()
    assert Path(path).exists()


def test_load_default_config():
    config = load_default_config()
    assert isinstance(config, dict)
    assert "panels" in config
    assert "paths" in config
    assert "train" in config
    assert "detect" in config
    # Paths are anchored to the package root, not the process cwd (monorepo / CI safe).
    proc = Path(config["paths"]["processed_dir"])
    assert proc.is_absolute()
    assert proc.name == "processed"
    assert (proc.parent / "raw").as_posix() == Path(config["paths"]["raw_dir"]).as_posix()


def test_load_config_train_section():
    config = load_default_config()
    train = config["train"]
    assert "train_ratio" in train
    assert "model" in train
    assert "lags" in train
    assert isinstance(train["train_ratio"], float)


def test_load_config_detect_section():
    config = load_default_config()
    detect = config["detect"]
    assert "scorer_name" in detect
    assert "high_quantile" in detect
    assert detect["scorer_name"] in ("KMeansScorer", "NormScorer", "IsolationForestScorer")


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/path/config.yaml"))
