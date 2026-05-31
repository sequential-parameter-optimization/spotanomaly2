"""Command-line interface for event detection system."""

import argparse
import sys
import warnings
from pathlib import Path

from spotanomaly2.application.config import get_default_config_path, load_config
from spotanomaly2.application.pipeline import Pipeline
from spotanomaly2.dashboard import LiveMonitor
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
        elif args.command == "live":
            monitor = LiveMonitor(config, logger)
            if args.interval:
                # Continuous monitoring mode
                return monitor.run_monitoring(args.interval)
            else:
                # Single run mode
                monitor.run_once()
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
