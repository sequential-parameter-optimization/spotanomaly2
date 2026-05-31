"""File storage operations for Parquet files."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def make_yaml_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for YAML/JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [make_yaml_serializable(x) for x in obj]
    return obj


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to Parquet file with compression."""
    ensure_dir(path.parent)
    df.to_parquet(path, compression="snappy", index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def load_panel_parquet(path: Path, panel_id: str) -> pd.DataFrame:
    """Load panel data from Parquet file."""
    file_path = path / f"panel_{panel_id}.parquet"
    return load_parquet(file_path)


def save_panel_parquet(df: pd.DataFrame, path: Path, panel_id: str) -> None:
    """Save panel data to Parquet file."""
    file_path = path / f"panel_{panel_id}.parquet"
    save_parquet(df, file_path)


def generate_timestamp() -> str:
    """Generate timestamp string for model versioning.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_timestamped_dir(base_dir: Path, model_timestamp: Optional[str] = None) -> Path:
    """Find the latest timestamped directory or a specific one.

    Args:
        base_dir: Base directory containing timestamped subdirectories
        model_timestamp: Optional specific timestamp to use

    Returns:
        Path to the timestamped directory

    Raises:
        FileNotFoundError: If no timestamped directories are found
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # If specific timestamp provided, use that directory
    if model_timestamp:
        timestamp_dir = base_dir / model_timestamp
        if not timestamp_dir.exists():
            raise FileNotFoundError(f"Timestamped directory not found: {timestamp_dir}")
        return timestamp_dir

    # Otherwise, find all timestamped directories
    # Pattern: YYYYMMDD_HHMMSS
    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

    timestamped_dirs = [d for d in base_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)]

    if not timestamped_dirs:
        raise FileNotFoundError(f"No timestamped directories found in: {base_dir}")

    # Sort by directory name (timestamp) descending and return the most recent
    timestamped_dirs.sort(key=lambda d: d.name, reverse=True)
    latest_dir = timestamped_dirs[0]

    return latest_dir


def find_latest_model(models_dir: Path, model_filename: str, model_timestamp: Optional[str] = None) -> Path:
    """Find the latest model file in a timestamped directory.

    Args:
        models_dir: Base models directory containing timestamped subdirectories
        model_filename: Model filename (e.g., "fc_model_panel_1.pkl")
        model_timestamp: Optional specific timestamp to use

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model file or directory not found
    """
    # If specific timestamp provided, use that directory
    if model_timestamp:
        timestamp_dir = models_dir / model_timestamp
        if not timestamp_dir.exists():
            raise FileNotFoundError(f"Timestamped directory not found: {timestamp_dir}")
        model_path = timestamp_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    # Otherwise, search through timestamped directories in reverse chronological order
    # until we find one that contains the model file
    if not models_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {models_dir}")

    # Pattern: YYYYMMDD_HHMMSS
    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

    timestamped_dirs = [d for d in models_dir.iterdir() if d.is_dir() and timestamp_pattern.match(d.name)]

    if not timestamped_dirs:
        raise FileNotFoundError(f"No timestamped directories found in: {models_dir}")

    # Sort by directory name (timestamp) descending and search for model file
    timestamped_dirs.sort(key=lambda d: d.name, reverse=True)

    for timestamp_dir in timestamped_dirs:
        model_path = timestamp_dir / model_filename
        if model_path.exists():
            return model_path

    # If we get here, no model file was found in any directory
    raise FileNotFoundError(f"Model file not found: {model_filename} in any timestamped directory in {models_dir}")


def save_raw_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Save raw data metadata to JSON file.

    Args:
        metadata: Dictionary containing metadata (start_date, end_date, etc.)
        path: Path to the metadata JSON file
    """
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_raw_metadata(path: Path) -> dict[str, Any]:
    """Load raw data metadata from JSON file.

    Args:
        path: Path to the metadata JSON file

    Returns:
        Dictionary containing metadata

    Raises:
        FileNotFoundError: If metadata file not found
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_panel_parquet_versioned(raw_dir: Path, panel_id: str, version: Optional[str] = None) -> pd.DataFrame:
    """Load panel data from versioned raw data directory.

    Args:
        raw_dir: Base raw data directory containing timestamped subdirectories
        panel_id: Panel identifier
        version: Optional specific version timestamp to use (defaults to latest)

    Returns:
        DataFrame with panel data

    Raises:
        FileNotFoundError: If panel file or directory not found
    """
    # Find the versioned directory
    version_dir = find_latest_timestamped_dir(raw_dir, version)

    # Load panel parquet from versioned directory
    return load_panel_parquet(version_dir, panel_id)
