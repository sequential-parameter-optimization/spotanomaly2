"""Live monitoring: incremental detection with an existing model, plus reporting.

``LiveMonitor`` is the dashboard-side counterpart of the batch ``Pipeline``. It
reuses the same domain/application services but is responsible only for live
prediction (incremental fetch → process → detect), the current-anomaly check,
HTML report generation, and (optionally) running the live report server in a
continuous monitoring loop.
"""

import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from spotanomaly2.application.data_manager import DataManager
from spotanomaly2.application.exogenous_downloader import ExogenousDownloadManager
from spotanomaly2.application.exogenous_joiner import ExogenousJoinManager
from spotanomaly2.application.readiness_checker import ReadinessChecker
from spotanomaly2.domain.anomaly_detector import AnomalyDetector
from spotanomaly2.domain.constants import LIVE_REPORT_SERVER_PORT, MIN_ROWS_FOR_DETECTION
from spotanomaly2.domain.primary_registry import resolve_primary_fetcher
from spotanomaly2.domain.processing.data_processor import DataProcessor
from spotanomaly2.infrastructure import logging, storage


class LiveMonitor:
    """Runs live anomaly detection against an already-trained model and reports it."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("LiveMonitor")
        self._data_manager = DataManager(config, self.logger)
        self._readiness_checker = ReadinessChecker(config, self.logger)
        self._exogenous_downloader = ExogenousDownloadManager(config, self.logger)
        self._exogenous_joiner = ExogenousJoinManager(config, self.logger)

    def run_once(self) -> dict[str, tuple]:
        """Live prediction: download new data, process, detect with existing model."""
        self.logger.info("=" * 60)
        self.logger.info("LIVE PREDICTION MODE")
        self.logger.info("=" * 60)

        if not self._readiness_checker.is_ready():
            error_msg = (
                "PREREQUISITE CHECK FAILED: Baseline data or trained models "
                "are missing.\n"
                "Live mode requires clean baseline data. Please run the full "
                "pipeline first (download → process → train → detect), e.g.:\n"
                "  pipeline.download(); pipeline.process(); pipeline.train(); pipeline.detect()\n"
                "This will fetch, process, and train models with complete data."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        panels = self.config["panels"]["panel_ids"]

        fetch_status: dict[str, dict] = {}

        # Step 1: Download and process new data
        self.logger.info("Step 1/2: Downloading and processing new data...")
        fetcher = resolve_primary_fetcher(self.config, self.logger)

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

        start, end = self._exogenous_downloader.derive_fetch_window(new_panel_data)
        self._exogenous_downloader.download_all(start, end, fetch_status)
        new_panel_data = self._exogenous_joiner.join_all(new_panel_data, fetch_status)
        processor = DataProcessor(self.config, self.logger)
        new_processed_data = processor.run(new_panel_data)

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

            sensor_cols = [c for c in df.columns if not c.startswith("exogenous_") and not c.endswith(weight_suffix)]
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
                from spotanomaly2.dashboard.report_generator import LiveReportGenerator

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

    def run_monitoring(self, interval_minutes: int) -> int:
        """Run live monitoring in a loop, serving the live HTML report.

        Args:
            interval_minutes: Time between predictions in minutes.

        Returns:
            Exit code (0 for success, non-zero for error).
        """
        if interval_minutes < 1:
            self.logger.error("Interval must be at least 1 minute")
            return 1

        interval_seconds = interval_minutes * 60
        iteration = 0
        running = True

        def signal_handler(signum, frame):
            nonlocal running
            self.logger.info(f"\nReceived signal {signum}. Shutting down gracefully...")
            running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._start_report_server()

        self.logger.info("=" * 70)
        self.logger.info("LIVE MONITORING STARTED")
        self.logger.info("=" * 70)
        self.logger.info(f"Interval: {interval_minutes} minutes")
        self.logger.info("Press Ctrl+C to stop")
        self.logger.info("=" * 70)

        while running:
            iteration += 1
            start_time = datetime.now()

            try:
                self.logger.info("")
                self.logger.info("=" * 70)
                self.logger.info(f"ITERATION {iteration} - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info("=" * 70)

                self.run_once()

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                self.logger.info(f"Iteration {iteration} completed in {duration:.1f} seconds")

            except KeyboardInterrupt:
                self.logger.info("\nKeyboard interrupt received. Stopping...")
                break
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
                self.logger.info("Continuing to next iteration...")

            # Wait for next iteration
            if running:
                next_run = time.time() + interval_seconds
                while running and time.time() < next_run:
                    remaining = int(next_run - time.time())
                    if remaining <= 0:
                        break

                    # Show countdown every 30 seconds or for last 10 seconds
                    if remaining % 30 == 0 or remaining <= 10:
                        mins, secs = divmod(remaining, 60)
                        self.logger.info(f"Next iteration in {mins}m {secs}s...")

                    # Sleep in small increments for quick shutdown
                    time.sleep(min(1, remaining))

        self.logger.info("=" * 70)
        self.logger.info("LIVE MONITORING STOPPED")
        self.logger.info(f"Total iterations completed: {iteration}")
        self.logger.info("=" * 70)

        return 0

    def _start_report_server(self) -> None:
        """Start the live report server in a background daemon thread (best-effort)."""
        if not self.config.get("report", {}).get("enabled", True):
            return
        try:
            from spotanomaly2.dashboard.report_server import LiveReportServer

            results_dir = Path(self.config["paths"]["results_dir"]) / "live"
            results_dir.mkdir(parents=True, exist_ok=True)

            server = LiveReportServer(results_dir, port=LIVE_REPORT_SERVER_PORT, logger=self.logger)

            def run_server():
                import asyncio

                try:
                    asyncio.run(server.start())
                except Exception as e:
                    self.logger.error(f"Server error: {e}", exc_info=True)

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            self.logger.info(
                f"Live report server listening on 0.0.0.0:{LIVE_REPORT_SERVER_PORT} (all network interfaces)"
            )
            self.logger.info(
                f"Open http://localhost:{LIVE_REPORT_SERVER_PORT} on this machine, "
                "or this host's IP from another device"
            )
        except Exception as e:
            self.logger.warning(f"Could not start live report server: {e}")
            self.logger.info("Continuing without live server (reports will still be generated)")
