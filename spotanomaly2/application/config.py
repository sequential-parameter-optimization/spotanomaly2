"""Configuration loading and management."""

from pathlib import Path
from typing import Any

import yaml

# The bundled default config lives at ``<repo>/config/default.yaml`` — that is,
# alongside the ``spotanomaly2`` package, not inside it. Resolving relative to
# this source file is robust to ``cwd`` and works for editable installs.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path_value(value: str, base: Path) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())


def _resolve_config_paths(config: dict[str, Any], base: Path) -> None:
    """Turn relative path entries in *config* into absolute paths under *base* (in place)."""
    paths = config.get("paths")
    if isinstance(paths, dict):
        for key in (
            "raw_dir",
            "processed_dir",
            "models_dir",
            "results_dir",
            "evaluations_dir",
            "credentials_file",
        ):
            val = paths.get(key)
            if isinstance(val, str):
                paths[key] = _resolve_path_value(val, base)

    process = config.get("process")
    if isinstance(process, dict):
        weather = process.get("weather")
        if isinstance(weather, dict):
            cp = weather.get("cache_path")
            if isinstance(cp, str):
                weather["cache_path"] = _resolve_path_value(cp, base)

    exogenous = config.get("exogenous")
    if isinstance(exogenous, dict):
        cd = exogenous.get("cache_dir")
        if isinstance(cd, str):
            exogenous["cache_dir"] = _resolve_path_value(cd, base)

    tune = config.get("tune")
    if isinstance(tune, dict):
        od = tune.get("output_dir")
        if isinstance(od, str):
            tune["output_dir"] = _resolve_path_value(od, base)

    train = config.get("train")
    if isinstance(train, dict):
        # Per-panel YAML paths under train.channel_config_files are read at training
        # time and rewritten by ModelTuner.update_channel_configs after tuning. Both
        # callers do `Path(value)` and join against cwd if relative — which breaks
        # when the kernel/CLI starts outside the repo root.
        channel_files = train.get("channel_config_files")
        if isinstance(channel_files, dict):
            for key, val in list(channel_files.items()):
                if isinstance(val, str):
                    channel_files[key] = _resolve_path_value(val, base)


def load_config(config_path: Path, base_dir: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.
        base_dir: Directory to resolve relative data/model paths against.
                  Defaults to the current working directory.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if base_dir is None:
        base_dir = Path.cwd()

    _resolve_config_paths(config, Path(base_dir))
    return config


def get_default_config_path() -> Path:
    """Get path to the bundled default configuration file (``config/default.yaml``)."""
    return _REPO_ROOT / "config" / "default.yaml"


def load_default_config(base_dir: Path | None = None) -> dict[str, Any]:
    """Load the bundled default configuration.

    Args:
        base_dir: Directory to resolve relative data/model paths against.
                  Defaults to the current working directory.
    """
    return load_config(get_default_config_path(), base_dir=base_dir)
