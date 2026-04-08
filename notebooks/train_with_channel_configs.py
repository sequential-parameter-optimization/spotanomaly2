"""Training script that uses channel-specific configurations.

This script is a convenience wrapper around Pipeline.train_panel() that allows
training a single panel with channel-specific configurations. It maintains
backward compatibility with the original standalone script interface while
using the same patterns and logic as the main Pipeline class.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from spotanomaly2.application.config import load_config
from spotanomaly2.application.pipeline import Pipeline
from spotanomaly2.infrastructure import logging


def main():
    """Run training with channel-specific configurations."""
    parser = argparse.ArgumentParser(description="Train models with channel-specific configs")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to base config file")
    parser.add_argument("--config-dir", type=str, default="config", help="Directory containing tuned config files")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing processed panel data")
    parser.add_argument("--panel", type=str, required=True, help="Panel ID to train (e.g., '1' or '2')")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = logging.get_logger("TrainingScript")
    logger.info(f"Starting training for panel {args.panel} with channel-specific configs...")
    
    # Load base config
    base_config = load_config(Path(args.config))
    config_dir = Path(args.config_dir)
    
    # Validate panel ID
    panel_ids = base_config.get("panels", {}).get("panel_ids", [])
    if args.panel not in panel_ids:
        logger.error(f"Invalid panel ID: {args.panel}. Available panels: {panel_ids}")
        return
    
    # Load panel data
    data_dir = Path(args.data_dir)
    panel_file = data_dir / f"panel_{args.panel}.parquet"
    
    if not panel_file.exists():
        logger.error(f"Panel file not found: {panel_file}")
        return
        
    logger.info(f"Loading panel {args.panel} from {panel_file}")
    panel_df = pd.read_parquet(panel_file)
    logger.info(f"Loaded panel {args.panel} with shape {panel_df.shape}")
    logger.info(f"Channels: {list(panel_df.columns)}")
    
    # Create Pipeline instance (uses same patterns as main CLI)
    pipeline = Pipeline(base_config, logger=logger)
    
    # Train using Pipeline's train_panel method
    # This automatically handles:
    # - Loading panel-level and channel-specific configs
    # - Mapping sanitized channel names to actual column names
    # - Training with appropriate hyperparameters
    # - Saving models and evaluation results
    logger.info("\nStarting training...")
    eval_df, timestamp = pipeline.train_panel(
        panel_id=args.panel,
        df=panel_df,
        config_dir=config_dir
    )
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("Training completed successfully!")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("="*80)
    logger.info("\nEvaluation Metrics:")
    logger.info(eval_df.to_string())
    logger.info("="*80)


if __name__ == "__main__":
    main()
