"""Test configuration loading."""

from pathlib import Path

import pytest

from spotanomaly2.application.config import (
    get_default_config_path,
    load_config,
    load_default_config,
    load_panel_channel_config,
)


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
    assert "split" in train
    assert "fallback_model" in train
    assert "lags" in train
    split = train["split"]
    assert {"train", "test", "score"} <= set(split)
    assert split["train"] + split["test"] + split["score"] == 100


def test_load_config_detect_section():
    config = load_default_config()
    detect = config["detect"]
    assert "scorer_name" in detect
    assert "high_quantile" in detect
    assert detect["scorer_name"] in ("KMeansScorer", "NormScorer", "IsolationForestScorer")


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/path/config.yaml"))


# ----- load_panel_channel_config --------------------------------------------


def test_load_panel_channel_config_returns_empty_when_no_train_section():
    assert load_panel_channel_config("1", {}) == {}


def test_load_panel_channel_config_returns_empty_when_file_map_missing():
    assert load_panel_channel_config("1", {"train": {}}) == {}


def test_load_panel_channel_config_returns_empty_when_file_map_not_a_dict():
    assert load_panel_channel_config("1", {"train": {"channel_config_files": "oops"}}) == {}


def test_load_panel_channel_config_returns_empty_when_panel_unknown(tmp_path):
    existing = tmp_path / "panel_other.yaml"
    existing.write_text("default:\n  model: LightGBM\n")
    config = {"train": {"channel_config_files": {"other": str(existing)}}}
    assert load_panel_channel_config("missing", config) == {}


def test_load_panel_channel_config_reads_yaml(tmp_path):
    cfg_path = tmp_path / "panel_1.yaml"
    cfg_path.write_text("default:\n  model: LightGBM\nchannels:\n  ch_a:\n    model: Ridge\n")
    config = {"train": {"channel_config_files": {"1": str(cfg_path)}}}

    loaded = load_panel_channel_config("1", config)
    assert loaded == {
        "default": {"model": "LightGBM"},
        "channels": {"ch_a": {"model": "Ridge"}},
    }


def test_load_panel_channel_config_falls_back_to_panel_prefixed_key(tmp_path):
    """When key 'X' is absent, fall back to 'panel_X' for backward compat."""
    cfg_path = tmp_path / "panel_1.yaml"
    cfg_path.write_text("default:\n  model: LightGBM\n")
    config = {"train": {"channel_config_files": {"panel_1": str(cfg_path)}}}

    assert load_panel_channel_config("1", config) == {"default": {"model": "LightGBM"}}


def test_load_panel_channel_config_empty_yaml_returns_empty_dict(tmp_path):
    cfg_path = tmp_path / "panel_1.yaml"
    cfg_path.write_text("")
    config = {"train": {"channel_config_files": {"1": str(cfg_path)}}}

    assert load_panel_channel_config("1", config) == {}


def test_load_panel_channel_config_missing_file_raises(tmp_path):
    config = {"train": {"channel_config_files": {"1": str(tmp_path / "nope.yaml")}}}
    with pytest.raises(FileNotFoundError):
        load_panel_channel_config("1", config)


def test_load_panel_channel_config_non_mapping_yaml_raises(tmp_path):
    cfg_path = tmp_path / "panel_1.yaml"
    cfg_path.write_text("- item1\n- item2\n")  # a list, not a mapping
    config = {"train": {"channel_config_files": {"1": str(cfg_path)}}}

    with pytest.raises(ValueError, match="must be a mapping"):
        load_panel_channel_config("1", config)
