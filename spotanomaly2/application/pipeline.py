"""Pipeline orchestration for event detection system."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.application.readiness_checker import ReadinessChecker
from spotanomaly2.domain.anomaly_detector import AnomalyDetector
from spotanomaly2.domain.constants import MIN_ROWS_FOR_DETECTION
from spotanomaly2.domain.data_processor import DataProcessor
from spotanomaly2.domain.exogenous_fetcher import ExogenousFetcher
from spotanomaly2.domain.model_trainer import ModelTrainer
from spotanomaly2.domain.primary_fetcher import PrimaryDataFetcher
from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp, make_yaml_serializable


class Pipeline:
    """Orchestrates the event detection pipeline (download, process, train, detect)."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("Pipeline")
        self._data_manager = DataManager(config, self.logger)
        self._readiness_checker = ReadinessChecker(config, self.logger)

    def is_ready(self, max_age_days: float = 7) -> bool:
        """Return True iff baseline processed data and trained models are present and fresh."""
        return self._readiness_checker.is_ready(max_age_days)

    # ------------------------------------------------------------------
    # Individual pipeline steps
    # ------------------------------------------------------------------

    def download(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("STEP: Download data from API")
        self.logger.info("=" * 60)

        primary_fetcher = PrimaryDataFetcher(self.config, self.logger)
        panel_data = primary_fetcher.run()
        self._data_manager.save_raw_data(panel_data)

    def process(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("STEP: Process data")
        self.logger.info("=" * 60)

        panel_data = self._data_manager.load_raw_data()
        panel_data = self._merge_exogenous_data(panel_data)
        processor = DataProcessor(self.config, self.logger)
        processed_data = processor.run(panel_data)
        self._data_manager.save_processed_data(processed_data)

    def train_panel(
        self,
        panel_id: str,
        df: pd.DataFrame,
        training_timestamp: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Train a single panel via spotforecast2."""
        trainer = ModelTrainer(self.config, self.logger)

        if training_timestamp is None:
            training_timestamp = generate_timestamp()

        eval_df, timestamp = trainer.train_panel(panel_id, df, timestamp=training_timestamp, save_model=True)

        base_models_dir = Path(self.config["paths"]["models_dir"])
        models_dir = base_models_dir / str(timestamp)
        eval_filename = f"fc_model_panel_{panel_id}_eval.csv"
        eval_path = models_dir / eval_filename
        eval_df.to_csv(eval_path)
        self.logger.info(f"Saved evaluation results to {eval_path}")

        return eval_df, timestamp

    def train(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("STEP: Train forecasting models")
        self.logger.info("=" * 60)

        panel_data = self._data_manager.load_processed_data()
        training_timestamp = generate_timestamp()

        for panel_id, df in panel_data.items():
            self.train_panel(panel_id=panel_id, df=df, training_timestamp=training_timestamp)

    def tune(
        self,
        panel_id: str | None = None,
        channel: str | None = None,
    ) -> dict:
        """Tune forecaster hyperparameters per channel per panel via SpotOptim.

        Args:
            panel_id: If provided, tune only this panel. Otherwise tune all panels.
            channel: If provided, tune only this channel within selected panel(s).

        Returns:
            Dict of panel_id -> channel -> tuning results.
        """
        import yaml

        self.logger.info("=" * 60)
        self.logger.info("STEP: Tune forecaster hyperparameters (SpotOptim)")
        self.logger.info("=" * 60)

        from spotanomaly2.domain.model_tuner import ModelTuner

        panel_data = self._data_manager.load_processed_data()

        if panel_id is not None:
            if panel_id not in panel_data:
                raise ValueError(f"Panel {panel_id} not found. Available: {list(panel_data.keys())}")
            panel_data = {panel_id: panel_data[panel_id]}

        channels_filter = [channel] if channel else None

        tuner = ModelTuner(self.config, self.logger)
        all_results = tuner.tune_all_panels(panel_data, channels=channels_filter)

        tune_cfg = self.config.get("tune", {})
        output_dir = Path(tune_cfg.get("output_dir", "data/tuning_results"))
        timestamp = storage.generate_timestamp()
        results_dir = output_dir / timestamp
        storage.ensure_dir(results_dir)

        for pid, channel_results in all_results.items():
            output_data = {
                "panel_id": pid,
                "timestamp": timestamp,
                "channels": {},
            }
            for ch_name, ch_result in channel_results.items():
                output_data["channels"][ch_name] = make_yaml_serializable(ch_result)

            result_path = results_dir / f"panel_{pid}.yaml"
            with open(result_path, "w") as f:
                yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Saved tuning results to {result_path}")

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

        tuner.update_channel_configs(all_results)

        return all_results

    def detect(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("STEP: Detect anomalies")
        self.logger.info("=" * 60)

        panel_data = self._data_manager.load_processed_data()
        detector = AnomalyDetector(self.config, self.logger)
        results = detector.run(panel_data)
        self._data_manager.save_detection_results(results)

    # ------------------------------------------------------------------
    # Composite runs
    # ------------------------------------------------------------------

    def run_all(self, skip_download: bool = False, predict_only: bool = False) -> None:
        if not predict_only:
            # Step 1: Download (or load from disk)
            if not skip_download:
                fetcher = PrimaryDataFetcher(self.config, self.logger)
                panel_data = fetcher.run()
                self._data_manager.save_raw_data(panel_data)
            else:
                self.logger.info("Skipping download step - loading from disk")
                panel_data = self._data_manager.load_raw_data()

            # Step 2: Process
            panel_data = self._merge_exogenous_data(panel_data)
            processor = DataProcessor(self.config, self.logger)
            processed_data = processor.run(panel_data)
            self._data_manager.save_processed_data(processed_data)

            # Step 3: Train
            trainer = ModelTrainer(self.config, self.logger)
            training_timestamp = generate_timestamp()
            eval_results = {}

            for panel_id, df in processed_data.items():
                eval_df, timestamp = trainer.train_panel(panel_id, df, timestamp=training_timestamp, save_model=True)
                eval_results[panel_id] = (eval_df, timestamp)

            base_models_dir = Path(self.config["paths"]["models_dir"])

            for panel_id, (eval_df, timestamp) in eval_results.items():
                models_dir = base_models_dir / str(timestamp)
                eval_filename = f"fc_model_panel_{panel_id}_eval.csv"
                eval_path = models_dir / eval_filename
                eval_df.to_csv(eval_path)
                self.logger.info(f"Saved evaluation results to {eval_path}")
        else:
            self.logger.info("Predict-only mode: skipping download, process, and train steps")
            processed_data = self._data_manager.load_processed_data()

        # Step 4: Detect
        detector = AnomalyDetector(self.config, self.logger)
        results = detector.run(processed_data)
        self._data_manager.save_detection_results(results)

        self.logger.info("=" * 60)
        self.logger.info("Pipeline completed successfully")
        self.logger.info("=" * 60)

    def live(self) -> dict[str, tuple]:
        """Live prediction: download new data, process, detect with existing model."""
        self.logger.info("=" * 60)
        self.logger.info("LIVE PREDICTION MODE")
        self.logger.info("=" * 60)

        if not self.is_ready():
            error_msg = (
                "PREREQUISITE CHECK FAILED: Baseline data or trained models "
                "are missing.\n"
                "Live mode requires clean baseline data. Please run the full "
                "pipeline first:\n"
                "  spotanomaly2 --config <your_config.yaml> all\n"
                "Or from Python:\n"
                "  pipeline.run_all()\n"
                "This will fetch, process, and train models with complete data."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        processed_dir = Path(self.config["paths"]["processed_dir"])
        panels = self.config["panels"]["panel_ids"]

        fetch_status: dict[str, dict] = {}

        # Step 1: Download and process new data
        self.logger.info("Step 1/2: Downloading and processing new data...")
        fetcher = PrimaryDataFetcher(self.config, self.logger)

        try:
            new_panel_data = fetcher.run(incremental_only=True)
            nan_panels = [pid for pid, df in new_panel_data.items() if df.empty or df.dropna(how="all").empty]
            if nan_panels:
                fetch_status["primary"] = {
                    "status": "degraded",
                    "error": f"NaN-only data for panel(s): {nan_panels}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.logger.warning(f"primary returned NaN-only data for panels: {nan_panels}")
            else:
                fetch_status["primary"] = {
                    "status": "ok",
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as exc:
            self.logger.error(f"primary fetch failed: {exc}", exc_info=True)
            fetch_status["primary"] = {
                "status": "error",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            new_panel_data = {pid: pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp")) for pid in panels}

        new_panel_data = self._merge_exogenous_data(new_panel_data, fetch_status)
        processor = DataProcessor(self.config, self.logger)
        new_processed_data = processor.run(new_panel_data)

        # Infer OpenMeteo status from processed output
        weather_enabled = self.config.get("process", {}).get("weather", {}).get("enabled", False)
        if weather_enabled:
            has_weather = any(
                any(c.startswith("weather_") for c in df.columns) for df in new_processed_data.values() if not df.empty
            )
            has_temperature = any("temperature" in df.columns for df in new_processed_data.values() if not df.empty)
            if has_weather or has_temperature:
                fetch_status["openmeteo"] = {
                    "status": "ok",
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                fetch_status["openmeteo"] = {
                    "status": "error",
                    "error": "Weather columns missing from processed data",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        else:
            fetch_status["openmeteo"] = {
                "status": "disabled",
                "error": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self._data_manager.save_processed_data_live(new_processed_data)

        complete_processed_data = self._data_manager.load_processed_data_live()
        processed_dir = Path(self.config["paths"]["processed_dir"])
        for panel_id, df in list(complete_processed_data.items()):
            if len(df) < MIN_ROWS_FOR_DETECTION:
                try:
                    full_df = storage.load_panel_parquet(processed_dir, panel_id)
                    if len(full_df) >= MIN_ROWS_FOR_DETECTION:
                        complete_processed_data[panel_id] = full_df
                        self.logger.info(
                            f"Panel {panel_id}: live data has {len(df)} rows; "
                            f"using full processed ({len(full_df)} rows) for this run"
                        )
                except FileNotFoundError:
                    pass

        # Detect data freshness issues per panel
        weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")
        freq_str = self.config.get("process", {}).get("resample", {}).get("freq", "5min")
        freq_td = pd.tseries.frequencies.to_offset(freq_str)
        now_utc = pd.Timestamp.now(tz="UTC")

        panel_nan_gaps: dict[str, dict] = {}
        for panel_id, df in complete_processed_data.items():
            if df.empty:
                panel_nan_gaps[panel_id] = {
                    "trailing_nan_rows": 0,
                    "trailing_nan_duration": None,
                    "last_valid_timestamp": None,
                    "latest_timestamp": None,
                    "data_age_seconds": None,
                }
                continue

            sensor_cols = [c for c in df.columns if not c.startswith("weather_") and not c.endswith(weight_suffix)]
            if not sensor_cols:
                continue

            sensor_df = df[sensor_cols]
            all_nan_mask = sensor_df.isna().all(axis=1)

            trailing_nan_count = 0
            for val in reversed(all_nan_mask.values):
                if val:
                    trailing_nan_count += 1
                else:
                    break

            non_nan = sensor_df.dropna(how="all")
            last_valid_idx = non_nan.index.max() if not non_nan.empty else None
            latest_idx = df.index.max()

            duration_str = None
            if trailing_nan_count > 0 and freq_td is not None:
                duration = trailing_nan_count * freq_td
                duration_str = str(duration)

            # How old is the latest valid sensor data relative to now?
            data_age_seconds = None
            if last_valid_idx is not None:
                ts = pd.Timestamp(last_valid_idx)
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                data_age_seconds = (now_utc - ts).total_seconds()

            panel_nan_gaps[panel_id] = {
                "trailing_nan_rows": trailing_nan_count,
                "trailing_nan_duration": duration_str,
                "last_valid_timestamp": (last_valid_idx.isoformat() if last_valid_idx is not None else None),
                "latest_timestamp": (latest_idx.isoformat() if latest_idx is not None else None),
                "data_age_seconds": data_age_seconds,
            }

            if trailing_nan_count > 0:
                self.logger.warning(
                    f"Panel {panel_id}: trailing NaN gap of {trailing_nan_count} "
                    f"rows ({duration_str}) - last valid data at {last_valid_idx}"
                )
            if data_age_seconds is not None and data_age_seconds > 3600:
                hours = data_age_seconds / 3600
                self.logger.warning(
                    f"Panel {panel_id}: sensor data is {hours:.1f} h old (last valid: {last_valid_idx})"
                )

        if "primary" in fetch_status:
            fetch_status["primary"]["panel_nan_gaps"] = panel_nan_gaps

            # Upgrade status to degraded if data is stale (>1 hour old) or has
            # long NaN tails
            has_stale = any((g.get("data_age_seconds") or 0) > 3600 for g in panel_nan_gaps.values())
            has_nan_tail = any(g["trailing_nan_rows"] >= 12 for g in panel_nan_gaps.values())
            if (has_stale or has_nan_tail) and fetch_status["primary"]["status"] == "ok":
                fetch_status["primary"]["status"] = "degraded"
                if has_stale:
                    fetch_status["primary"]["error"] = "Sensor data is stale (see per-panel details)"
                else:
                    fetch_status["primary"]["error"] = "Sensor data contains long NaN gaps (see per-panel details)"

        # Step 2: Detect anomalies
        self.logger.info("Step 2/2: Detecting anomalies with existing model...")
        detector = AnomalyDetector(self.config, self.logger)
        results = detector.run(complete_processed_data)

        self._check_current_anomalies(results)

        results_dir = self._data_manager.save_detection_results(results, live_mode=True)

        report_enabled = self.config.get("report", {}).get("enabled", True)
        if report_enabled:
            try:
                self.logger.info("Generating HTML report...")
                from spotanomaly2.domain.report_generator import LiveReportGenerator

                report_generator = LiveReportGenerator(self.config, self.logger)
                report_path = report_generator.generate_report(
                    results_dir=results_dir,
                    panel_ids=list(results.keys()),
                    processed_data=complete_processed_data,
                    fetch_status=fetch_status,
                )
                self.logger.info(f"HTML report saved to: {report_path}")
            except Exception as e:
                self.logger.error(f"Error generating HTML report: {e}", exc_info=True)
                self.logger.warning("Continuing without HTML report...")

        self.logger.info("=" * 60)
        self.logger.info("Live prediction completed successfully")
        self.logger.info("=" * 60)

        return results

    def _check_current_anomalies(self, results: dict[str, tuple]) -> None:
        """Check and report anomalies at the current (latest) timestamp."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("CURRENT ANOMALY CHECK")
        self.logger.info("=" * 70)

        any_anomalies = False

        for panel_id, (scores_df, flags_df, forecast_df, _contrib, *_rest) in results.items():
            if len(flags_df) == 0:
                continue

            latest_timestamp = flags_df.index.max()
            latest_flags = flags_df.loc[latest_timestamp]
            anomaly_channels = latest_flags[latest_flags > 0].index.tolist()

            if anomaly_channels:
                any_anomalies = True
                self.logger.warning(f"   ANOMALY DETECTED - Panel {panel_id} at {latest_timestamp}")
                self.logger.warning(f"   Affected channels: {', '.join(anomaly_channels)}")
                latest_scores = scores_df.loc[latest_timestamp]
                for channel in anomaly_channels:
                    if channel in latest_scores.index:
                        score = latest_scores[channel]
                        self.logger.warning(f"   - {channel}: score = {score:.4f}")
            else:
                self.logger.info(f"  Panel {panel_id} at {latest_timestamp}: No anomalies")

        if not any_anomalies:
            self.logger.info("")
            self.logger.info("  All clear - No anomalies detected at current timestamp")

        self.logger.info("=" * 70)

    def _merge_exogenous_data(
        self,
        panel_data: dict[str, pd.DataFrame],
        fetch_status: dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch Exogenous timeseries and merge into every panel DataFrame."""
        if not self.config.get("exogenous", {}).get("enabled", False):
            if fetch_status is not None:
                fetch_status["exogenous"] = {
                    "status": "disabled",
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return panel_data

        try:
            fetcher = ExogenousFetcher(self.config, self.logger)
            result = fetcher.merge_into_panels(panel_data)
            if fetch_status is not None:
                fetch_status["exogenous"] = {
                    "status": "ok",
                    "error": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return result
        except Exception as exc:
            self.logger.warning(f"Exogenous fetch failed, continuing without Exogenous data: {exc}")
            if fetch_status is not None:
                fetch_status["exogenous"] = {
                    "status": "error",
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return panel_data