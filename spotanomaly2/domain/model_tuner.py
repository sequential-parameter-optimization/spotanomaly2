"""Model tuning service using spotforecast2 SpotOptim."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from spotanomaly2.domain.spotforecast_adapter import SpotforecastTuner
from spotanomaly2.infrastructure import logging


def _normalize_lag_spec(value: Any) -> str:
    """Normalize a lag specification to a string for SpotOptim categorical factors.

    SpotOptim treats lag candidates as categorical ``"factor"`` variables and
    uses ``parse_lags_from_strings`` internally to convert them back.  All
    candidates in the list must therefore be **strings** so that SpotOptim's
    ``process_factor_bounds`` receives homogeneous factor levels.

    Handles YAML values that may already be strings (``"48"``,
    ``"[1, 2, 24]"``), plain ints, or native lists.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return str([int(v) for v in value])
    return str(int(value))


def _build_lag_candidates(shared_lags: list[Any] | None, global_lags: Any) -> list[str]:
    """Build lag candidates from config with stable de-duplication.

    Uses configured ``tune.lags`` as the primary source. If not provided,
    falls back to ``train.lags``.
    """
    ordered_raw: list[Any] = []
    if shared_lags:
        ordered_raw.extend(shared_lags)
    elif global_lags is not None:
        ordered_raw.append(global_lags)

    seen: set[str] = set()
    normalized: list[str] = []
    for value in ordered_raw:
        value_str = _normalize_lag_spec(value)
        if value_str in seen:
            continue
        seen.add(value_str)
        normalized.append(value_str)

    return normalized


class ModelTuner:
    """Merges global/panel/channel tune config and delegates to SpotforecastTuner."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("ModelTuner")
        self.tuner = SpotforecastTuner(config, self.logger)

    def _build_tune_config(self) -> dict[str, Any]:
        """Build the effective tune config, merging global defaults with panel overrides."""
        tune_section = self.config.get("tune", {})
        models_raw = tune_section.get("models", {"LightGBM": {}})

        if not isinstance(models_raw, dict):
            raise ValueError(
                "Invalid tune.models configuration: expected a mapping of "
                "model_name -> search_space, for example: "
                "tune.models.LightGBM: {...}"
            )

        models = list(models_raw.keys())
        model_search_spaces = {k: dict(v or {}) for k, v in models_raw.items()}

        # Shared lags are injected into each model's search space.
        # Normalize all lag specs to strings so SpotOptim treats them as
        # homogeneous categorical factor levels (it parses them back
        # internally via parse_lags_from_strings).
        shared_lags = tune_section.get("lags")
        global_lags = self.config.get("train", {}).get("lags", 6)
        lag_candidates = _build_lag_candidates(shared_lags, global_lags)
        for space in model_search_spaces.values():
            space.setdefault("lags", lag_candidates)

        return {
            "n_trials": tune_section.get("n_trials", 10),
            "n_initial": tune_section.get("n_initial", 5),
            "metric": tune_section.get("metric", "mean_absolute_error"),
            "models": models,
            "model_search_spaces": model_search_spaces,
            "panel_overrides": tune_section.get("panel_overrides", {}),
        }

    def tune_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        tune_config = self._build_tune_config()
        return self.tuner.tune_panel(panel_id, df, tune_config, channels=channels)

    def update_channel_configs(self, all_results: dict[str, dict[str, dict[str, Any]]]) -> None:
        """Update channel config YAML files with tuning results."""
        channel_config_files = self.config.get("train", {}).get("channel_config_files", {})
        base_dir = Path(self.config.get("_config_base_dir", Path.cwd()))

        for pid, channel_results in all_results.items():
            cfg_path_value = channel_config_files.get(pid) or channel_config_files.get(f"panel_{pid}")
            if not cfg_path_value:
                continue

            cfg_path = Path(cfg_path_value)
            if not cfg_path.is_absolute():
                cfg_path = (base_dir / cfg_path).resolve()

            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    panel_cfg = yaml.safe_load(f) or {}
            else:
                panel_cfg = {}

            channels_section = panel_cfg.setdefault("channels", {})

            for ch_name, ch_result in channel_results.items():
                if "error" in ch_result:
                    continue
                best_lags = ch_result.get("best_lags")
                if best_lags is None:
                    continue
                if hasattr(best_lags, "tolist"):
                    best_lags = best_lags.tolist()
                elif isinstance(best_lags, (list, tuple)):
                    best_lags = [int(x) for x in best_lags]
                else:
                    best_lags = int(best_lags)

                best_params = ch_result.get("best_params", {})
                # Convert numpy types to native Python for YAML
                clean_params = {}
                for k, v in best_params.items():
                    if isinstance(v, (np.integer,)):
                        v = int(v)
                    elif isinstance(v, (np.floating,)):
                        v = float(v)
                    clean_params[k] = v

                channels_section[ch_name] = {
                    "model": ch_result.get("best_model"),
                    "params": clean_params,
                    "best_lags": best_lags,
                }

            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(panel_cfg, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Updated channel config: {cfg_path}")

    def tune_all_panels(
        self,
        panel_data: dict[str, pd.DataFrame],
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        results = {}
        panel_ids = list(panel_data.keys())
        panel_pbar = tqdm(panel_ids, desc="Panels", unit="panel", leave=True)
        for panel_id in panel_pbar:
            panel_pbar.set_postfix_str(f"panel {panel_id}")
            results[panel_id] = self.tune_panel(panel_id, panel_data[panel_id], channels=channels)
        panel_pbar.close()
        return results
