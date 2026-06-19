"""Configuration loading and management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataSplit:
    """Disjoint train / test / score percentages of the processed data.

    Three contiguous windows that partition the data 100%:

    - ``train``: forecaster fits here.
    - ``test``: trainer's eval window AND tuner's CV val window. Held out from
      the forecaster's fit so the trainer-reported RMSE/MAE and the tuner's
      leaderboard are out-of-sample.
    - ``score``: scorer's territory. **Never** seen by the trainer or tuner —
      this is what closes the hyperparameter-selection leakage the older
      ``train_ratio`` design had.

    Values are integer percentages and must sum to 100.
    """

    train: int
    test: int
    score: int

    def __post_init__(self) -> None:
        total = self.train + self.test + self.score
        if total != 100:
            raise ValueError(
                f"train.split percentages must sum to 100, got "
                f"train={self.train} + test={self.test} + score={self.score} = {total}"
            )
        if min(self.train, self.test, self.score) <= 0:
            raise ValueError(
                f"train.split percentages must all be positive, got "
                f"train={self.train}, test={self.test}, score={self.score}"
            )


def resolve_data_split(config: dict[str, Any]) -> DataSplit:
    """Read ``config['train']['split']`` into a :class:`DataSplit` (validates sum)."""
    raw = config.get("train", {}).get("split", {})
    return DataSplit(
        train=int(raw.get("train", 80)),
        test=int(raw.get("test", 10)),
        score=int(raw.get("score", 10)),
    )


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
            "tuning_dir",
            "results_dir",
            "evaluations_dir",
            "credentials_file",
        ):
            val = paths.get(key)
            if isinstance(val, str):
                paths[key] = _resolve_path_value(val, base)

    # exogenous is now a list of {name, fetcher, joiner, config} entries.
    # Resolve cache_dir / cache_path inside each entry's config block against base_dir.
    exogenous = config.get("exogenous")
    if isinstance(exogenous, list):
        for entry in exogenous:
            if not isinstance(entry, dict):
                continue
            src_cfg = entry.get("config")
            if not isinstance(src_cfg, dict):
                continue
            for path_key in ("cache_dir", "cache_path"):
                val = src_cfg.get(path_key)
                if isinstance(val, str):
                    src_cfg[path_key] = _resolve_path_value(val, base)

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
    # Validate train.split sums to 100. Done at load so a misconfigured split
    # surfaces immediately, not deep inside the trainer/tuner.
    resolve_data_split(config)
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


def load_panel_channel_config(panel_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Read the per-panel channel-model YAML referenced by ``train.channel_config_files``.

    Paths in ``train.channel_config_files`` are pre-resolved to absolute by
    :func:`load_config`; this helper assumes that and does not re-resolve.

    Args:
        panel_id: Panel identifier. The key ``panel_<panel_id>`` is checked as a fallback
            for backward compatibility with older config layouts.
        config: Configuration dictionary (post-``load_config``).

    Returns:
        Parsed YAML mapping for the panel, or an empty dict if no file is configured.

    Raises:
        FileNotFoundError: The configured path does not exist.
        ValueError: The YAML content is not a mapping.
    """
    file_map = config.get("train", {}).get("channel_config_files", {})
    if not isinstance(file_map, dict):
        return {}

    cfg_path_value = file_map.get(panel_id) or file_map.get(f"panel_{panel_id}")
    if not cfg_path_value:
        return {}

    cfg_path = Path(cfg_path_value)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Channel model config for panel {panel_id} not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Channel model config for panel {panel_id} must be a mapping: {cfg_path}")
    return loaded
