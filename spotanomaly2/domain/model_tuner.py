"""Model tuning service using spotforecast2 SpotOptim."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from spotanomaly2.application.config import load_panel_channel_config
from spotanomaly2.domain.spotforecast_adapter import SpotforecastTuner
from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import make_yaml_serializable


class ModelTuner:
    """Merges global/panel/channel tune config and delegates to SpotforecastTuner."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("ModelTuner")
        self.tuner = SpotforecastTuner(config, self.logger)

    def _build_tune_config(self, panel_id: str | None = None) -> dict[str, Any]:
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

        # Shared lags are injected into each model's search space
        shared_lags = tune_section.get("lags")
        if shared_lags:
            for space in model_search_spaces.values():
                space.setdefault("lags", shared_lags)

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
        tune_config = self._build_tune_config(panel_id)
        return self.tuner.tune_panel(panel_id, df, tune_config, channels=channels)

    def update_channel_configs(self, all_results: dict[str, dict[str, dict[str, Any]]]) -> None:
        """Update channel config YAML files with tuning results."""
        channel_config_files = self.config.get("train", {}).get("channel_config_files", {})

        for pid, channel_results in all_results.items():
            cfg_path_value = channel_config_files.get(pid) or channel_config_files.get(f"panel_{pid}")
            if not cfg_path_value:
                self.logger.warning(
                    f"Panel {pid}: tuning succeeded but no train.channel_config_files entry — "
                    f"results not persisted to channel YAML (raw tuning_results YAML still saved)"
                )
                continue
            cfg_path = Path(cfg_path_value)

            try:
                panel_cfg = load_panel_channel_config(pid, self.config)
            except FileNotFoundError:
                # First tune run for this panel; file will be created below.
                panel_cfg = {}

            channels_section = panel_cfg.setdefault("channels", {})

            for ch_name, ch_result in channel_results.items():
                if "error" in ch_result:
                    continue
                best_lags = ch_result.get("best_lags")
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

    def _resolve_output_dir(self) -> Path:
        """Return the directory for tuning result YAMLs.

        Precedence: ``paths.tuning_dir`` (preferred, sibling of
        ``paths.models_dir``) > legacy ``tune.output_dir`` > default.
        """
        paths_cfg = self.config.get("paths", {}) or {}
        tune_cfg = self.config.get("tune", {}) or {}
        return Path(paths_cfg.get("tuning_dir") or tune_cfg.get("output_dir") or "data/tuning_results")

    def _persist_results(
        self,
        all_results: dict[str, dict[str, dict[str, Any]]],
        results_dir: Path,
    ) -> None:
        """Dump one ``panel_<id>.yaml`` per panel under ``results_dir``."""
        storage.ensure_dir(results_dir)
        for pid, channel_results in all_results.items():
            output_data = {
                "panel_id": pid,
                "timestamp": results_dir.name,
                "channels": {
                    ch_name: make_yaml_serializable(ch_result) for ch_name, ch_result in channel_results.items()
                },
            }
            result_path = results_dir / f"panel_{pid}.yaml"
            with open(result_path, "w") as f:
                yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Saved tuning results to {result_path}")

    def _log_summary(
        self,
        all_results: dict[str, dict[str, dict[str, Any]]],
        results_dir: Path,
    ) -> None:
        """Log the human-readable per-channel winner leaderboard."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("TUNING RESULTS SUMMARY")
        self.logger.info("=" * 70)
        for pid, channel_results in all_results.items():
            self.logger.info(f"\nPanel {pid}:")
            self.logger.info("-" * 50)
            for ch_name, ch_result in channel_results.items():
                if "error" in ch_result:
                    self.logger.error(f"  {ch_name}: FAILED - {ch_result['error']}")
                    continue
                metric_val = ch_result.get("best_metric")
                best_lags = ch_result.get("best_lags")
                best_model = ch_result.get("best_model", "?")
                best_params = ch_result.get("best_params", {})
                self.logger.info(f"  {ch_name}:")
                self.logger.info(f"    model  = {best_model}")
                self.logger.info(f"    metric = {metric_val}")
                self.logger.info(f"    lags   = {best_lags}")
                for pname, pval in best_params.items():
                    self.logger.info(f"    {pname} = {pval}")
        self.logger.info("=" * 70)
        self.logger.info(f"Results saved to: {results_dir}")
        self.logger.info("=" * 70)

    def run(
        self,
        panel_data: dict[str, pd.DataFrame],
        channels: list[str] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Full tune workflow: search → persist YAML → summary log → channel config update.

        Mirrors how ``SpotforecastTrainer`` owns its own artifact persistence
        (the trainer writes its ``.pkl`` itself), so ``Pipeline.tune`` stays a
        thin orchestrator instead of doing I/O and presentation work.
        """
        all_results = self.tune_all_panels(panel_data, channels=channels)

        results_dir = self._resolve_output_dir() / storage.generate_timestamp()
        self._persist_results(all_results, results_dir)
        self._log_summary(all_results, results_dir)
        self.update_channel_configs(all_results)

        return all_results
