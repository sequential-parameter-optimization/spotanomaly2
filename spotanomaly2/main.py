"""Command-line interface for event detection system."""

import argparse
import signal
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

from spotanomaly2.application.config import get_default_config_path, load_config
from spotanomaly2.application.pipeline import Pipeline
from spotanomaly2.domain.constants import LIVE_REPORT_SERVER_PORT
from spotanomaly2.infrastructure import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*TimeSeries with tz.*")
warnings.filterwarnings("ignore", message=".*Failed to set device parameter.*")
warnings.filterwarnings("ignore", message=".*One-step-ahead predictions are used.*")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="event_detection", description="Improved forecast-based anomaly detection system"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Helper function to add config argument to any parser
    def add_config_argument(subparser):
        subparser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="Path to configuration file (default: config/default.yaml)",
        )

    # Download command
    download_parser = subparsers.add_parser("download", help="Download data from API and save as Parquet")
    add_config_argument(download_parser)

    # Process command
    process_parser = subparsers.add_parser("process", help="Process raw data (convert, resample, preprocess)")
    add_config_argument(process_parser)
    process_parser.add_argument(
        "--raw-data-version",
        type=str,
        default=None,
        help=("Raw data version timestamp (e.g. 20260105_174531). Default: most recent version."),
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train forecasting models")
    add_config_argument(train_parser)
    train_parser.add_argument(
        "--raw-data-version",
        type=str,
        default=None,
        help=("Raw data version timestamp (e.g. 20260105_174531). Default: most recent version."),
    )

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect anomalies using trained models")
    add_config_argument(detect_parser)
    detect_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model timestamp to use (e.g., 20250115_143022). If not provided, uses the most recent model.",
    )
    detect_parser.add_argument(
        "--raw-data-version",
        type=str,
        default=None,
        help=("Raw data version timestamp (e.g. 20260105_174531). Default: most recent version."),
    )

    # All command
    all_parser = subparsers.add_parser("all", help="Run all steps sequentially")
    add_config_argument(all_parser)
    all_parser.add_argument("--skip-download", action="store_true", help="Skip the download step")
    all_parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and only run detection (uses most recent model)",
    )
    all_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model timestamp to use (e.g., 20250115_143022). If not provided, uses the most recent model.",
    )
    all_parser.add_argument(
        "--raw-data-version",
        type=str,
        default=None,
        help=("Raw data version timestamp (e.g. 20260105_174531). Default: most recent version."),
    )

    # Tune command
    tune_parser = subparsers.add_parser(
        "tune", help="Tune forecaster hyperparameters per channel per panel using SpotOptim"
    )
    add_config_argument(tune_parser)
    tune_parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help="Tune only this panel ID (e.g., '1'). If not provided, tunes all panels.",
    )
    tune_parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Tune only this channel column (e.g., 'channel_1_ph'). If not provided, tunes all channels.",
    )
    tune_parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of SpotOptim trials (overrides config value).",
    )
    tune_parser.add_argument(
        "--n-initial",
        type=int,
        default=None,
        help="Override number of initial random evaluations (overrides config value).",
    )

    # Live command
    live_parser = subparsers.add_parser(
        "live",
        help="Live prediction: download new data, process, and predict with existing model (no training)",
    )
    add_config_argument(live_parser)
    live_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model timestamp to use (e.g., 20250115_143022). If not provided, uses the most recent model.",
    )
    live_parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Run continuously with this interval in minutes. If not provided, runs once and exits.",
    )

    return parser


def run_live_monitoring(pipeline: Pipeline, interval_minutes: int, logger) -> int:
    """Run live monitoring in a loop.

    Args:
        pipeline: Pipeline instance
        interval_minutes: Time between predictions in minutes
        logger: Logger instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if interval_minutes < 1:
        logger.error("Interval must be at least 1 minute")
        return 1

    interval_seconds = interval_minutes * 60
    iteration = 0
    running = True
    server_thread = None

    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"\nReceived signal {signum}. Shutting down gracefully...")
        running = False

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start live report server in background
    report_enabled = pipeline.config.get("report", {}).get("enabled", True)
    if report_enabled:
        try:
            from spotanomaly2.infrastructure.report_server import LiveReportServer

            results_dir = Path(pipeline.config["paths"]["results_dir"]) / "live"
            results_dir.mkdir(parents=True, exist_ok=True)

            server = LiveReportServer(results_dir, port=LIVE_REPORT_SERVER_PORT, logger=logger)

            def run_server():
                import asyncio

                try:
                    asyncio.run(server.start())
                except Exception as e:
                    logger.error(f"Server error: {e}", exc_info=True)

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            logger.info(f"Live report server listening on 0.0.0.0:{LIVE_REPORT_SERVER_PORT} (all network interfaces)")
            logger.info(
                f"Open http://localhost:{LIVE_REPORT_SERVER_PORT} on this machine, "
                "or this host's IP from another device"
            )
        except Exception as e:
            logger.warning(f"Could not start live report server: {e}")
            logger.info("Continuing without live server (reports will still be generated)")

    logger.info("=" * 70)
    logger.info("LIVE MONITORING STARTED")
    logger.info("=" * 70)
    logger.info(f"Interval: {interval_minutes} minutes")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 70)

    while running:
        iteration += 1
        start_time = datetime.now()

        try:
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"ITERATION {iteration} - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            pipeline.live()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Iteration {iteration} completed in {duration:.1f} seconds")

        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received. Stopping...")
            break
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
            logger.info("Continuing to next iteration...")

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
                    logger.info(f"Next iteration in {mins}m {secs}s...")

                # Sleep in small increments for quick shutdown
                time.sleep(min(1, remaining))

    logger.info("=" * 70)
    logger.info("LIVE MONITORING STOPPED")
    logger.info(f"Total iterations completed: {iteration}")
    logger.info("=" * 70)

    return 0


def main(argv=None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Command-line arguments (for testing)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = get_default_config_path()

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Preserve the directory the config file came from so other components
    # can resolve relative paths consistently.
    if args.config:
        config_base_dir = Path(config_path).resolve().parent
    else:
        config_base_dir = Path.cwd()
    config["_config_base_dir"] = str(config_base_dir)

    # Add model_timestamp to config if provided via CLI
    if hasattr(args, "model") and args.model:
        if "detect" not in config:
            config["detect"] = {}
        config["detect"]["model_timestamp"] = args.model

    # Add raw_data_version to config if provided via CLI
    if hasattr(args, "raw_data_version") and args.raw_data_version:
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["raw_data_version"] = args.raw_data_version

    # Get logger
    logger = logging.get_logger()

    try:
        # Create pipeline instance
        pipeline = Pipeline(config, logger)

        # Execute command
        if args.command == "download":
            pipeline.download()
        elif args.command == "process":
            pipeline.process()
        elif args.command == "train":
            pipeline.train()
        elif args.command == "detect":
            pipeline.detect()
        elif args.command == "tune":
            if args.n_trials is not None:
                config.setdefault("tune", {})["n_trials"] = args.n_trials
            if args.n_initial is not None:
                config.setdefault("tune", {})["n_initial"] = args.n_initial
            pipeline.tune(panel_id=args.panel, channel=args.channel)
        elif args.command == "all":
            pipeline.run_all(skip_download=args.skip_download, predict_only=args.predict_only)
        elif args.command == "live":
            if args.interval:
                # Continuous monitoring mode
                return run_live_monitoring(pipeline, args.interval, logger)
            else:
                # Single run mode
                pipeline.live()
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
