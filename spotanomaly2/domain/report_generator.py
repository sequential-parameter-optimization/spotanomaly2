"""HTML report generator for live anomaly detection results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spotanomaly2.infrastructure import logging


def convert_numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays and pandas types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy arrays or pandas types

    Returns:
        Object with numpy arrays converted to lists
    """
    # Handle numpy arrays first
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy scalars
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle datetime/timestamp types - check by type name as well for robustness
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    elif isinstance(obj, np.timedelta64):
        return str(obj)
    # Check by type name for any Timestamp-like objects
    elif type(obj).__name__ in ("Timestamp", "DatetimeTZDtype"):
        try:
            return pd.Timestamp(obj).isoformat()
        except (ValueError, TypeError):
            return str(obj)
    # Handle collections
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


# Color scheme matching anomaly_scores_with_background.qmd
COLORS = {
    "background": "#090909",  # Very dark background
    "border": "#A90F12",  # Bright red border
    "contents": "rgba(169, 15, 18, 0.55)",  # Stronger transparent red anomaly region fill
    "line": "#5EA6E1",  # Light blue line
    "anomaly_marker": "#A90F12",  # Red anomaly markers
    "anomaly_marker_edge": "#5EA6E1",  # Blue edge
}


def detect_anomaly_regions(df: pd.DataFrame, max_gap_minutes: int = 60) -> list[dict]:
    """
    Detect continuous regions where anomalies occur close together.
    Anomalies within max_gap_minutes are considered part of the same region.

    Parameters:
    -----------
    df : DataFrame
        DataFrame with anomaly_flag column (timestamp as index)
    max_gap_minutes : int
        Maximum gap in minutes between anomalies to be considered same region

    Returns:
    --------
    list of dicts
        List of regions with start_time and end_time
    """
    if len(df) == 0:
        return []

    # Get anomaly timestamps
    anomaly_times = df[df["anomaly_flag"] == 1].index

    if len(anomaly_times) == 0:
        return []

    regions = []
    region_start = anomaly_times[0]
    region_end = anomaly_times[0]

    max_gap = pd.Timedelta(minutes=max_gap_minutes)

    for i in range(1, len(anomaly_times)):
        current_time = anomaly_times[i]
        gap = current_time - region_end

        if gap <= max_gap:
            # Extend current region
            region_end = current_time
        else:
            # Save current region and start new one
            # Extend region boundaries slightly for visual clarity
            padding = pd.Timedelta(minutes=10)
            regions.append(
                {
                    "start": max(df.index[0], region_start - padding),
                    "end": min(df.index[-1], region_end + padding),
                }
            )
            region_start = current_time
            region_end = current_time

    # Add the last region
    padding = pd.Timedelta(minutes=10)
    regions.append(
        {
            "start": max(df.index[0], region_start - padding),
            "end": min(df.index[-1], region_end + padding),
        }
    )

    return regions


class LiveReportGenerator:
    """Generate HTML reports for live anomaly detection results."""

    def __init__(self, config: dict[str, Any], logger=None):
        """Initialize report generator.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.get_logger("LiveReportGenerator")
        self.lookback_days = config.get("report", {}).get("lookback_days", 14)
        self.output_filename = config.get("report", {}).get("output_filename", "report.html")
        self.refresh_interval = config.get("report", {}).get("refresh_interval", 30)
        self.timezone_name = str(config.get("report", {}).get("timezone", "local"))
        self._timezone = self._resolve_timezone(self.timezone_name)

        # Load provider display names from config
        self.primary_name = config.get("primary", {}).get("display_name", "Primary")
        self.exogenous_name = self._first_non_weather_display_name() or "Exogenous"

    def _exogenous_source_config(self, source_name: str) -> dict:
        """Return the ``config:`` sub-block for a named entry in ``config['exogenous']``."""
        for entry in self.config.get("exogenous", []) or []:
            if isinstance(entry, dict) and entry.get("name") == source_name:
                return entry.get("config", {}) or {}
        return {}

    def _first_non_weather_display_name(self) -> str | None:
        for entry in self.config.get("exogenous", []) or []:
            if not isinstance(entry, dict) or entry.get("name") == "weather":
                continue
            src_cfg = entry.get("config", {}) or {}
            if "display_name" in src_cfg:
                return src_cfg["display_name"]
        return None

    def _resolve_timezone(self, timezone_name: str):
        """Resolve configured timezone name to a tzinfo object.

        "local" uses the machine local timezone.
        """
        if timezone_name.lower() == "local":
            return datetime.now().astimezone().tzinfo

        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            self.logger.warning(f"Invalid report.timezone '{timezone_name}', falling back to local timezone")
            return datetime.now().astimezone().tzinfo

    def _to_report_timezone_index(self, idx: pd.Index) -> pd.DatetimeIndex:
        """Convert a datetime-like index to report timezone and drop tz info for plotting."""
        dt_idx = pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
        dt_idx = dt_idx.tz_convert(self._timezone)
        return dt_idx.tz_localize(None)

    def _apply_report_timezone_to_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with DatetimeIndex converted to report timezone."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        df_out = df.copy()
        df_out.index = self._to_report_timezone_index(df_out.index)
        return df_out

    def _get_channel_source_label(self, panel_id: str, column_name: str) -> str:
        """Infer data source label for a plotted channel column."""
        if column_name.startswith("weather_"):
            return "Open-Meteo"

        if column_name == "temperature":
            weather_cfg = self._exogenous_source_config("weather")
            lat = weather_cfg.get("latitude")
            lon = weather_cfg.get("longitude")
            place = weather_cfg.get("place") or weather_cfg.get("location")

            if lat is not None and lon is not None:
                if place:
                    location_part = f"{place}, {lat:.4f}, {lon:.4f}"
                else:
                    location_part = f"{lat:.4f}, {lon:.4f}"
            else:
                location_part = "configured weather location"

            return (
                "Derived (mean of panel temperature sensors minus Open-Meteo "
                f"temperature_2m rolling baseline at {location_part})"
            )

        if column_name.startswith("exogenous_"):
            return self.exogenous_name

        panels_cfg = self.config.get("panels", {})
        instrumentation_ids = panels_cfg.get("instrumentation_ids", {})
        channel_keys = panels_cfg.get("channel_keys", {})

        panel_instrumentation = instrumentation_ids.get(str(panel_id), {})

        for channel_id, keys in channel_keys.items():
            if not isinstance(keys, list):
                continue

            for key in keys:
                expected_name = f"channel_{channel_id}_{key}"
                if column_name == expected_name:
                    inst_id = panel_instrumentation.get(str(channel_id))
                    if inst_id is not None:
                        return f"{self.primary_name} (Instrumentation {inst_id})"
                    return self.primary_name

        if column_name.startswith("channel_"):
            return self.primary_name

        return "Derived/Unknown"

    def _extract_current_status(
        self,
        panel_id: str,
        df_scores: pd.DataFrame,
        df_flags: pd.DataFrame,
        df_actual: pd.DataFrame,
        df_forecast: pd.DataFrame,
    ) -> dict:
        """Extract status of the last 12 datapoints.

        Args:
            panel_id: Panel identifier
            df_scores: DataFrame with anomaly scores
            df_flags: DataFrame with anomaly flags
            df_actual: DataFrame with actual values
            df_forecast: DataFrame with forecast values; its columns define the
                set of scorable target channels (exogenous and weight columns
                in ``df_actual`` are filtered out via column intersection).

        Returns:
            Dictionary with current status information including last 12 points
        """
        status = {
            "panel_id": panel_id,
            "timestamp": None,
            "has_anomaly": False,
            "anomaly_channels": [],
            "score": None,
            "score_normalized": None,
            "all_values": {},
            "recent_points": [],  # Last 12 datapoints with anomaly status
        }

        if len(df_scores) == 0:
            return status

        # Get the last 12 timestamps (or fewer if not enough data)
        n_points = min(12, len(df_scores))
        recent_times = df_scores.index[-n_points:]

        # Build list of recent points
        for ts in recent_times:
            point = {
                "timestamp": ts.isoformat() if ts else None,
                "has_anomaly": False,
                "score": None,
                "score_normalized": None,
            }

            # Check anomaly flag
            if "anomaly_flag" in df_flags.columns and ts in df_flags.index:
                flag = df_flags.loc[ts, "anomaly_flag"]
                point["has_anomaly"] = bool(flag == 1)

            # Get score
            if "anomaly_score" in df_scores.columns and ts in df_scores.index:
                score_val = df_scores.loc[ts, "anomaly_score"]
                point["score"] = float(score_val) if pd.notna(score_val) else None
            # Get normalized score (0-1) when available
            if "anomaly_score_normalized" in df_scores.columns and ts in df_scores.index:
                norm_val = df_scores.loc[ts, "anomaly_score_normalized"]
                point["score_normalized"] = float(norm_val) if pd.notna(norm_val) else None

            status["recent_points"].append(point)

        # Get the most recent timestamp for overall status
        latest_time = df_scores.index.max()
        status["timestamp"] = latest_time.isoformat() if latest_time else None

        # Check if there's an anomaly in the latest point
        if "anomaly_flag" in df_flags.columns:
            latest_flag = df_flags.loc[latest_time, "anomaly_flag"] if latest_time in df_flags.index else 0
            status["has_anomaly"] = bool(latest_flag == 1)

        # Get the anomaly score
        if "anomaly_score" in df_scores.columns and latest_time in df_scores.index:
            score_val = df_scores.loc[latest_time, "anomaly_score"]
            status["score"] = float(score_val) if pd.notna(score_val) else None
        # Get normalized score (0-1) when available
        if "anomaly_score_normalized" in df_scores.columns and latest_time in df_scores.index:
            norm_val = df_scores.loc[latest_time, "anomaly_score_normalized"]
            status["score_normalized"] = float(norm_val) if pd.notna(norm_val) else None

        # Get all channel values at latest time. Restrict to columns the model
        # actually scores (i.e. forecast targets) so exogenous_* and *__weight
        # columns merged into df_actual upstream are not reported as channels.
        channel_cols = df_actual.columns.intersection(df_forecast.columns)
        if latest_time in df_actual.index:
            latest_row = df_actual.loc[latest_time, channel_cols]
            for col in latest_row.index:
                val = latest_row[col]
                status["all_values"][col] = float(val) if pd.notna(val) else None

        # If anomaly, try to identify which channels contributed
        if status["has_anomaly"] and latest_time in df_flags.index:
            latest_flags = df_flags.loc[latest_time]
            for col in latest_flags.index:
                if col != "anomaly_flag" and latest_flags[col] == 1:
                    status["anomaly_channels"].append(col)

            # If no specific channel flags, list all channels as potentially contributing
            if not status["anomaly_channels"]:
                status["anomaly_channels"] = list(status["all_values"].keys())

            # Sort affected channels by their current value (ascending; None last)
            def _channel_sort_key(ch: str) -> tuple:
                v = status["all_values"].get(ch)
                if v is None:
                    return (1, 0.0)
                return (0, v)

            status["anomaly_channels"] = sorted(status["anomaly_channels"], key=_channel_sort_key, reverse=True)

        return status

    def generate_report(
        self,
        results_dir: Path,
        panel_ids: list[str],
        processed_data: dict[str, pd.DataFrame],
        fetch_status: dict[str, dict] | None = None,
    ) -> Path:
        """Generate HTML report for all panels.

        Args:
            results_dir: Directory containing prediction results CSVs
            panel_ids: List of panel IDs to include in report
            processed_data: Dictionary mapping panel_id to processed DataFrame
            fetch_status: Optional dict mapping source name to status info

        Returns:
            Path to generated HTML file
        """
        self.logger.info(f"Generating HTML report for panels: {panel_ids}")

        timestamp = datetime.now(self._timezone)

        # Load results for each panel
        figures = {}
        event_contributions: dict[str, list[dict[str, Any]]] = {}
        current_statuses = {}

        for panel_id in panel_ids:
            self.logger.info(f"Processing panel {panel_id}...")

            # Load CSVs
            scores_file = results_dir / f"panel_{panel_id}_scores.csv"
            flags_file = results_dir / f"panel_{panel_id}_flags.csv"
            forecast_file = results_dir / f"panel_{panel_id}_forecast.csv"

            if not all(f.exists() for f in [scores_file, flags_file, forecast_file]):
                self.logger.warning(f"Missing result files for panel {panel_id}, skipping...")
                continue

            df_scores = pd.read_csv(scores_file)
            df_flags = pd.read_csv(flags_file)
            df_forecast = pd.read_csv(forecast_file)

            # Convert timestamps
            for df in [df_scores, df_flags, df_forecast]:
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)

            # Get processed actual data
            df_actual = processed_data.get(panel_id, pd.DataFrame())
            if len(df_actual) == 0:
                self.logger.warning(f"No processed data for panel {panel_id}, skipping...")
                continue

            # Ensure index is datetime
            if not isinstance(df_actual.index, pd.DatetimeIndex):
                if "timestamp" in df_actual.columns:
                    df_actual.set_index("timestamp", inplace=True)
                df_actual.index = pd.to_datetime(df_actual.index)

            # Convert all relevant series to report timezone for consistent plotting.
            df_scores = self._apply_report_timezone_to_index(df_scores)
            df_flags = self._apply_report_timezone_to_index(df_flags)
            df_forecast = self._apply_report_timezone_to_index(df_forecast)
            df_actual = self._apply_report_timezone_to_index(df_actual)

            # Filter to last N days
            if len(df_scores) > 0:
                end_date = df_scores.index.max()
                start_date = end_date - pd.Timedelta(days=self.lookback_days)
                df_scores = df_scores[df_scores.index >= start_date]
                df_flags = df_flags[df_flags.index >= start_date]
                df_forecast = df_forecast[df_forecast.index >= start_date]
                df_actual = df_actual[df_actual.index >= start_date]

            # Align dataframes by index - use loc instead of reindex to avoid NaN values
            common_index = df_scores.index.intersection(df_forecast.index).intersection(df_actual.index)
            if len(common_index) == 0:
                self.logger.warning(f"No common timestamps for panel {panel_id}, skipping...")
                continue

            # Sort index to ensure proper ordering
            common_index = common_index.sort_values()

            # Use loc to select only common indices (avoids introducing NaN from reindex)
            df_scores = df_scores.loc[common_index]
            df_flags = df_flags.loc[common_index]
            df_forecast = df_forecast.loc[common_index]
            df_actual = df_actual.loc[common_index]

            # Verify we have valid data
            if df_actual.empty or df_forecast.empty:
                self.logger.warning(f"Empty dataframes after alignment for panel {panel_id}, skipping...")
                continue

            # Ensure indices still match after loc (they should)
            if not df_actual.index.equals(df_forecast.index):
                self.logger.warning(f"Index mismatch after alignment for panel {panel_id}, attempting to fix...")
                # Force alignment by using intersection again
                final_common = df_actual.index.intersection(df_forecast.index).sort_values()
                df_actual = df_actual.loc[final_common]
                df_forecast = df_forecast.loc[final_common]
                df_scores = df_scores.loc[final_common]
                df_flags = df_flags.loc[final_common]

            # Reset index to have timestamp as a column (like the notebook does)
            # This matches the notebook implementation which uses df_merged['timestamp']
            df_actual_for_plot = df_actual.reset_index()
            df_forecast_for_plot = df_forecast.reset_index()
            # Rename index column to 'timestamp' if it exists
            if df_actual_for_plot.columns[0] != "timestamp":
                df_actual_for_plot.rename(columns={df_actual_for_plot.columns[0]: "timestamp"}, inplace=True)
            if df_forecast_for_plot.columns[0] != "timestamp":
                df_forecast_for_plot.rename(columns={df_forecast_for_plot.columns[0]: "timestamp"}, inplace=True)

            # Extract current status for this panel
            try:
                current_statuses[panel_id] = self._extract_current_status(
                    panel_id, df_scores, df_flags, df_actual, df_forecast
                )
            except Exception as e:
                self.logger.error(f"Error extracting current status for panel {panel_id}: {e}", exc_info=True)

            # Create visualizations
            try:
                panel_age = None
                if fetch_status:
                    panel_age = (
                        fetch_status.get("primary", {})
                        .get("panel_nan_gaps", {})
                        .get(str(panel_id), {})
                        .get("data_age_seconds")
                    )
                figures[f"panel{panel_id}_normalized"] = self._create_normalized_scores_plot(
                    panel_id,
                    df_scores,
                    df_flags,
                    data_age_seconds=panel_age,
                )
                # Load scorer-aligned contributions if available
                contrib_file = results_dir / f"panel_{panel_id}_contributions.parquet"
                df_contributions = None
                if contrib_file.exists():
                    try:
                        df_contributions = pd.read_parquet(contrib_file)
                        df_contributions = self._apply_report_timezone_to_index(df_contributions)
                        df_contributions = df_contributions.loc[
                            df_contributions.index.intersection(common_index).sort_values()
                        ]
                        self.logger.info(
                            f"Panel {panel_id}: loaded scorer-aligned contributions ({len(df_contributions)} rows)"
                        )
                    except Exception as e:
                        self.logger.warning(f"Panel {panel_id}: failed to load contributions.parquet: {e}")
                        df_contributions = None

                panel_events = self._get_event_contributions(
                    panel_id,
                    df_actual,
                    df_forecast,
                    df_scores,
                    df_flags,
                    df_contributions=df_contributions,
                )
                # Reverse only event/region order (newest first) while keeping per-event feature order unchanged.
                event_contributions[panel_id] = list(reversed(panel_events))
                event_contrib_fig = self._create_event_contributions_plot(panel_id, event_contributions[panel_id])
                if event_contrib_fig is not None:
                    figures[f"panel{panel_id}_event_contributions"] = event_contrib_fig
                # Pass dataframes with timestamp as column for channels plot (like notebook)
                figures[f"panel{panel_id}_channels"] = self._create_channels_overview_plot(
                    panel_id,
                    df_actual_for_plot,
                    df_forecast_for_plot,
                    df_flags,
                    data_age_seconds=panel_age,
                )

                # --- Per-channel anomaly detection results ---
                pc_scores_file = results_dir / f"panel_{panel_id}_per_channel_scores.csv"
                pc_flags_file = results_dir / f"panel_{panel_id}_per_channel_flags.csv"
                pc_thresholds_file = results_dir / f"panel_{panel_id}_per_channel_thresholds.csv"
                pc_combined_file = results_dir / f"panel_{panel_id}_per_channel_flags_combined.csv"

                if all(f.exists() for f in [pc_scores_file, pc_flags_file, pc_thresholds_file, pc_combined_file]):
                    try:
                        df_pc_scores = pd.read_csv(pc_scores_file, index_col=0, parse_dates=True)
                        df_pc_flags = pd.read_csv(pc_flags_file, index_col=0, parse_dates=True)
                        df_pc_thresholds = pd.read_csv(pc_thresholds_file, index_col=0)
                        df_pc_combined = pd.read_csv(pc_combined_file, index_col=0, parse_dates=True)

                        # Apply report timezone
                        df_pc_scores = self._apply_report_timezone_to_index(df_pc_scores)
                        df_pc_flags = self._apply_report_timezone_to_index(df_pc_flags)
                        df_pc_combined = self._apply_report_timezone_to_index(df_pc_combined)

                        # Filter to lookback window
                        if len(df_pc_scores) > 0:
                            df_pc_scores = df_pc_scores[df_pc_scores.index >= start_date]
                            df_pc_flags = df_pc_flags[df_pc_flags.index >= start_date]
                            df_pc_combined = df_pc_combined[df_pc_combined.index >= start_date]

                        figures[f"panel{panel_id}_per_channel"] = self._create_per_channel_scores_plot(
                            panel_id,
                            df_pc_scores,
                            df_pc_flags,
                            df_pc_thresholds,
                            df_pc_combined,
                        )
                        self.logger.info(f"Panel {panel_id}: created per-channel scores plot")
                    except Exception as e:
                        self.logger.warning(f"Panel {panel_id}: failed to create per-channel plot: {e}")

            except Exception as e:
                self.logger.error(f"Error creating plots for panel {panel_id}: {e}", exc_info=True)
                continue

        # Generate HTML
        output_path = results_dir / self.output_filename

        # Copy logo asset to results_dir so the HTML can reference it by relative path
        _logo_src = Path(__file__).parent.parent.parent / "assets" / "spotlogo_red.png"
        _logo_dst = results_dir / "spotlogo_red.png"
        if _logo_src.exists() and not _logo_dst.exists():
            import shutil

            shutil.copy2(_logo_src, _logo_dst)

        # Save figure data and metadata to separate files for live updates
        data_path = results_dir / "figures.json"
        metadata_path = results_dir / "metadata.json"
        status_path = results_dir / "current_status.json"

        self._assemble_html(
            figures,
            event_contributions,
            timestamp,
            output_path,
            data_path,
            metadata_path,
            current_statuses,
            fetch_status=fetch_status,
        )

        # Save fetch status to a separate JSON file for live updates
        if fetch_status:
            fetch_status_path = results_dir / "fetch_status.json"
            with open(fetch_status_path, "w", encoding="utf-8") as f:
                json.dump(fetch_status, f, indent=2)

        # Save current status to JSON file LAST (after metadata.json)
        # This ensures file watcher only triggers once per update cycle
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(current_statuses, f, indent=2)

        self.logger.info(f"HTML report saved to: {output_path}")
        return output_path

    def _create_normalized_scores_plot(
        self,
        panel_id: str,
        df_scores: pd.DataFrame,
        df_flags: pd.DataFrame,
        data_age_seconds: float | None = None,
    ) -> go.Figure:
        """Create normalized scores plot with background highlighting.

        Args:
            panel_id: Panel identifier
            df_scores: DataFrame with anomaly scores (must have 'anomaly_score_normalized' column)
            df_flags: DataFrame with anomaly flags
            data_age_seconds: Seconds since last valid sensor reading (for NO DATA region)

        Returns:
            Plotly figure
        """
        # Merge flags with scores
        df_merged = df_scores.copy()
        if "anomaly_flag" not in df_merged.columns:
            df_merged["anomaly_flag"] = df_flags["anomaly_flag"] if "anomaly_flag" in df_flags.columns else 0

        # Ensure normalized scores exist
        if "anomaly_score_normalized" not in df_merged.columns:
            # Fallback: use regular scores normalized manually
            scores = df_merged["anomaly_score"].values
            if len(scores) > 0:
                q99 = np.quantile(scores, 0.99)
                q01 = np.quantile(scores, 0.01)
                if q99 > q01:
                    normalized = (scores - q01) / (q99 - q01)
                    normalized = np.clip(normalized, 0.0, 1.0)
                else:
                    normalized = np.full_like(scores, 0.5)
                df_merged["anomaly_score_normalized"] = normalized
            else:
                df_merged["anomaly_score_normalized"] = 0.0

        y_max = 1.0

        # Detect anomaly regions
        regions = detect_anomaly_regions(df_merged, max_gap_minutes=60)
        region_count = len(regions)

        # Create figure
        fig = go.Figure()

        # Add content rectangles (filled regions where anomalies occur)
        for region in regions:
            fig.add_shape(
                type="rect",
                x0=region["start"],
                x1=region["end"],
                y0=0,
                y1=y_max,
                fillcolor=COLORS["contents"],
                line=dict(width=0),
                layer="below",
            )

        # Add normalized score line
        # Convert to list to avoid binary serialization issues with numpy arrays
        fig.add_trace(
            go.Scatter(
                x=df_merged.index.tolist(),
                y=df_merged["anomaly_score_normalized"].tolist(),
                mode="lines",
                name="Normalized Score",
                line=dict(color=COLORS["line"], width=2.5),
                hovertemplate=f"<b>Panel {panel_id}</b><br>Time: %{{x}}<br>Normalized Score: %{{y:.4f}}<extra></extra>",
            )
        )

        # Add anomaly markers
        anomaly_mask = df_merged["anomaly_flag"] == 1
        if anomaly_mask.any():
            anomaly_timestamps = df_merged.index[anomaly_mask]
            anomaly_region_indices: list[int] = []

            for ts in anomaly_timestamps:
                region_idx = -1
                for idx, region in enumerate(regions):
                    if region["start"] <= ts <= region["end"]:
                        # Event/contribution panels are rendered in reversed region order.
                        # Store the reversed index so marker interaction maps to the visible order.
                        region_idx = (region_count - 1 - idx) if region_count > 0 else -1
                        break
                anomaly_region_indices.append(region_idx)

            # Convert to list to avoid binary serialization issues with numpy arrays
            fig.add_trace(
                go.Scatter(
                    x=anomaly_timestamps.tolist(),
                    y=df_merged["anomaly_score_normalized"][anomaly_mask].tolist(),
                    mode="markers",
                    name="Detected Anomaly",
                    customdata=anomaly_region_indices,
                    marker=dict(
                        color=COLORS["anomaly_marker"],
                        size=10,
                        symbol="circle",
                        line=dict(color=COLORS["anomaly_marker_edge"], width=1.5),
                    ),
                    hovertemplate="<b>[! ] Anomaly</b><br>Time: %{x}<br>Score: %{y:.4f}<br><i>Click again to unfocus | Esc clears all focus</i><extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(
            title={
                "text": f"Panel {panel_id}: Normalized Anomaly Scores<br><sub>Bordered regions show anomaly clusters</sub>",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "color": COLORS["line"]},
            },
            xaxis_title="Timestamp",
            yaxis_title="Normalized Anomaly Score",
            yaxis=dict(range=[0, 1], gridcolor="rgba(128, 128, 128, 0.15)", linecolor=COLORS["border"]),
            xaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)", linecolor=COLORS["border"]),
            height=600,
            hovermode="x unified",
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0, 0, 0, 0.5)",
                bordercolor=COLORS["border"],
                borderwidth=1,
            ),
        )

        # Draw a "No Data" shaded region when sensor data is stale
        if data_age_seconds is not None and data_age_seconds > 3600 and len(df_merged) > 0:
            data_end = df_merged.index.max()
            now_naive = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
            hours_missing = data_age_seconds / 3600
            gap_label = (
                f"{int(hours_missing // 24)}d {int(hours_missing % 24)}h"
                if hours_missing >= 24
                else f"{int(hours_missing)}h"
            )
            fig.add_vrect(
                x0=data_end,
                x1=now_naive,
                fillcolor="rgba(169, 15, 18, 0.25)",
                line=dict(width=0),
                layer="below",
            )
            fig.add_annotation(
                x=data_end + (now_naive - data_end) / 2,
                y=0.5,
                text=f"NO DATA ({gap_label})",
                showarrow=False,
                font=dict(size=14, color="#A90F12", family="monospace"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="#A90F12",
                borderwidth=1,
                borderpad=6,
            )

        return fig

    def _create_contribution_plot(
        self,
        panel_id: str,
        df_actual: pd.DataFrame,
        df_forecast: pd.DataFrame,
        df_scores: pd.DataFrame,
        df_contributions: pd.DataFrame | None = None,
    ) -> go.Figure:
        """Create score contribution estimation plot.

        When *df_contributions* (scorer-aligned, from ``explain()``) is
        available its fractional contributions are used directly.  Otherwise
        falls back to the z-score-of-actuals heuristic.

        Args:
            panel_id: Panel identifier
            df_actual: DataFrame with actual values
            df_forecast: DataFrame with forecast values
            df_scores: DataFrame with anomaly scores
            df_contributions: Optional scorer-aligned per-channel contributions

        Returns:
            Plotly figure
        """
        if df_contributions is not None and not df_contributions.empty:
            score_col = "anomaly_score" if "anomaly_score" in df_scores.columns else df_scores.columns[0]
            feature_score_contributions = df_contributions.mul(
                df_scores[score_col].reindex(df_contributions.index).values, axis=0
            )
            avg_score_contribution = feature_score_contributions.mean().sort_values(ascending=False)
        else:
            common_cols = df_actual.columns.intersection(df_forecast.columns)
            if len(common_cols) == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="No matching columns between actual and forecast data",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
                return fig

            df_residuals = df_actual[common_cols] - df_forecast[common_cols]
            feature_stds = df_actual[common_cols].std().replace(0, 1.0)
            df_normalized_residuals = df_residuals.div(feature_stds, axis=1)
            df_squared_normalized_residuals = df_normalized_residuals**2
            total_squared_residuals = df_squared_normalized_residuals.sum(axis=1).replace(0, np.nan)
            feature_contributions_pct = df_squared_normalized_residuals.div(total_squared_residuals, axis=0)

            score_col = (
                "anomaly_score"
                if "anomaly_score" in df_scores.columns
                else (
                    "anomaly_score_normalized"
                    if "anomaly_score_normalized" in df_scores.columns
                    else df_scores.columns[0]
                )
            )
            feature_score_contributions = feature_contributions_pct.mul(df_scores[score_col].values, axis=0)
            avg_score_contribution = feature_score_contributions.mean().sort_values(ascending=False)

        # Create horizontal bar chart
        fig = go.Figure()

        # Convert to list to avoid binary serialization issues with numpy arrays
        fig.add_trace(
            go.Bar(
                x=avg_score_contribution.values.tolist(),
                y=avg_score_contribution.index.tolist(),
                orientation="h",
                marker=dict(color=COLORS["line"]),
                hovertemplate="<b>%{y}</b><br>Average Contribution: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title={
                "text": f"Panel {panel_id}: Estimated Average Contribution to Anomaly Score by Feature",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18, "color": COLORS["line"]},
            },
            xaxis_title="Average Score Contribution",
            yaxis_title="Feature",
            height=max(400, len(avg_score_contribution) * 40),
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)", linecolor=COLORS["border"]),
            yaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)", linecolor=COLORS["border"]),
        )

        return fig

    def _get_event_contributions(
        self,
        panel_id: str,
        df_actual: pd.DataFrame,
        df_forecast: pd.DataFrame,
        df_scores: pd.DataFrame,
        df_flags: pd.DataFrame,
        top_n: int = 5,
        df_contributions: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Compute per-event feature contributions for detected anomalies.

        When *df_contributions* (scorer-aligned, from ``explain()``) is
        available its fractional contributions are used directly.  Otherwise
        falls back to the z-score-of-actuals heuristic.

        Args:
            panel_id: Panel identifier
            df_actual: DataFrame with actual values
            df_forecast: DataFrame with forecast values
            df_scores: DataFrame with anomaly scores
            df_flags: DataFrame with anomaly flags
            top_n: Number of top features to return per event
            df_contributions: Optional scorer-aligned per-channel contributions

        Returns:
            List of event contribution dicts
        """
        if "anomaly_score" in df_scores.columns:
            score_series = df_scores["anomaly_score"]
        else:
            score_col = (
                "anomaly_score_normalized" if "anomaly_score_normalized" in df_scores.columns else df_scores.columns[0]
            )
            score_series = df_scores[score_col]

        if df_contributions is not None and not df_contributions.empty:
            feature_score_contributions = df_contributions.mul(
                score_series.reindex(df_contributions.index).values, axis=0
            )
        else:
            common_cols = df_actual.columns.intersection(df_forecast.columns)
            if len(common_cols) == 0:
                self.logger.warning(f"Panel {panel_id}: No common columns for event contributions")
                return []

            df_residuals = df_actual[common_cols] - df_forecast[common_cols]
            feature_stds = df_actual[common_cols].std().replace(0, 1.0)
            df_normalized_residuals = df_residuals.div(feature_stds, axis=1)
            df_squared_normalized_residuals = df_normalized_residuals**2
            total_squared_residuals = df_squared_normalized_residuals.sum(axis=1).replace(0, np.nan)
            feature_contributions_pct = df_squared_normalized_residuals.div(total_squared_residuals, axis=0)
            feature_score_contributions = feature_contributions_pct.mul(score_series.values, axis=0)

        # Build anomaly flag series
        if "anomaly_flag" in df_flags.columns:
            anomaly_mask = df_flags["anomaly_flag"] == 1
        else:
            flag_col = df_flags.columns[0] if len(df_flags.columns) > 0 else None
            anomaly_mask = df_flags[flag_col] == 1 if flag_col else pd.Series(False, index=df_flags.index)

        if anomaly_mask.sum() == 0:
            return []

        # Cluster anomalies into regions (use existing logic)
        try:
            regions = detect_anomaly_regions(df_flags, max_gap_minutes=60)
        except Exception:
            regions = []

        if not regions:
            # Fallback to per-timestamp events
            event_times = df_flags.index[anomaly_mask]
            events = []
            for ts in event_times:
                if ts not in feature_score_contributions.index:
                    continue
                row = feature_score_contributions.loc[ts].dropna()
                if row.empty:
                    continue
                top_features = row.sort_values(ascending=False).head(top_n)
                score_value = score_series.loc[ts] if ts in score_series.index else None
                events.append(
                    {
                        "timestamp": ts,
                        "anomaly_score": score_value,
                        "top_features": [
                            {"feature": feature, "contribution": float(value)}
                            for feature, value in top_features.items()
                        ],
                    }
                )
            return events

        # Aggregate contributions per region
        events = []
        for region in regions:
            region_start = region.get("start")
            region_end = region.get("end")
            if region_start is None or region_end is None:
                continue

            region_mask = (feature_score_contributions.index >= region_start) & (
                feature_score_contributions.index <= region_end
            )
            region_df = feature_score_contributions.loc[region_mask]
            if region_df.empty:
                continue

            avg_contrib = region_df.mean().dropna()
            if avg_contrib.empty:
                continue

            top_features = avg_contrib.sort_values(ascending=False).head(top_n)
            region_score = score_series.loc[region_mask].mean() if region_mask.any() else None

            events.append(
                {
                    "region_start": region_start,
                    "region_end": region_end,
                    "anomaly_score": region_score,
                    "top_features": [
                        {"feature": feature, "contribution": float(value)} for feature, value in top_features.items()
                    ],
                }
            )

        return events

    def _create_event_contributions_plot(self, panel_id: str, events: list[dict[str, Any]]) -> go.Figure:
        """Create a plot for per-event (clustered) feature contributions.

        Args:
            panel_id: Panel identifier
            events: List of event contribution dicts

        Returns:
            Plotly figure
        """
        if not events:
            return None

        # Build per-event subplots to avoid overlap
        from plotly.subplots import make_subplots

        subplot_titles = []
        for idx, event in enumerate(events, start=1):
            if event.get("region_start") and event.get("region_end"):
                title = f"Region {idx}: {event['region_start']} -> {event['region_end']}"
            elif event.get("timestamp"):
                title = f"Event {idx}: {event['timestamp']}"
            else:
                title = f"Event {idx}"
            subplot_titles.append(title)

        n_event_rows = len(events)
        # Pixel-based sizing: Plotly's `vertical_spacing` is a fraction of the
        # *total* figure height, so for tall figures the gaps otherwise balloon
        # and compress bars to nothing. Convert a target pixel gap to a fraction.
        per_row_plot_px = 160
        # Gap must fit the upper subplot's x-axis ticks + "Contribution" title
        # plus the next subplot's title annotation — anything less collides.
        gap_px = 80
        extras_px = 100  # layout title + bottom axis label
        total_height = n_event_rows * per_row_plot_px + max(0, n_event_rows - 1) * gap_px + extras_px
        total_height = max(400, total_height)
        vertical_spacing = gap_px / total_height if n_event_rows > 1 else 0.0

        fig = make_subplots(
            rows=n_event_rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=vertical_spacing,
            subplot_titles=subplot_titles,
        )

        for idx, event in enumerate(events, start=1):
            top_features = event.get("top_features", [])
            if not top_features:
                continue

            # Keep per-event feature order consistent.
            features = [item["feature"] for item in reversed(top_features)]
            values = [item["contribution"] for item in reversed(top_features)]

            fig.add_trace(
                go.Bar(
                    x=values,
                    y=features,
                    orientation="h",
                    marker=dict(color=COLORS["line"]),
                    showlegend=False,
                    hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
                ),
                row=idx,
                col=1,
            )

        fig.update_layout(
            title={
                "text": f"Panel {panel_id}: Per-Event Feature Contributions",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18, "color": COLORS["line"]},
            },
            height=total_height,
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
        )

        for i in range(1, len(events) + 1):
            fig.update_xaxes(
                title_text="Contribution",
                gridcolor="rgba(128, 128, 128, 0.15)",
                linecolor=COLORS["border"],
                row=i,
                col=1,
            )
            fig.update_yaxes(
                title_text="Feature",
                gridcolor="rgba(128, 128, 128, 0.15)",
                linecolor=COLORS["border"],
                row=i,
                col=1,
            )

        return fig

    def _create_per_channel_scores_plot(
        self,
        panel_id: str,
        df_pc_scores: pd.DataFrame,
        df_pc_flags: pd.DataFrame,
        df_pc_thresholds: pd.DataFrame,
        df_pc_flags_combined: pd.DataFrame,
    ) -> go.Figure:
        """Create per-channel anomaly scores plot.

        Shows each channel's absolute residual as its own subplot. A top-level
        subplot shows the combined per-channel flag.

        Args:
            panel_id: Panel identifier
            df_pc_scores: Per-channel absolute residuals (n_eval, n_channels)
            df_pc_flags: Per-channel binary flags (unused; kept for caller compat)
            df_pc_thresholds: Fitted thresholds (unused; kept for caller compat)
            df_pc_flags_combined: Combined per-channel flag (n_eval, 1)

        Returns:
            Plotly figure
        """
        channels = df_pc_scores.columns.tolist()
        n_channels = len(channels)

        # First row = combined per-channel flag timeline, then one row per channel
        n_rows = 1 + n_channels
        subplot_titles = ["Per-Channel Anomaly (combined)"] + [c for c in channels]

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=[0.8] + [1.0] * n_channels,
        )

        # --- Row 1: combined per-channel flag as filled area (0/1) ---
        combined_vals = df_pc_flags_combined["per_channel_anomaly_flag"].tolist()
        fig.add_trace(
            go.Scatter(
                x=df_pc_flags_combined.index.tolist(),
                y=combined_vals,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(169, 15, 18, 0.35)",
                line=dict(color=COLORS["anomaly_marker"], width=1),
                name="Per-channel flag",
                showlegend=True,
                hovertemplate="<b>Per-Channel Flag</b><br>Time: %{x}<br>Flag: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["OK", "Anomaly"], row=1, col=1)

        # Palette for channels
        channel_colors = [
            "#5EA6E1",
            "#F5A623",
            "#7ED321",
            "#D0021B",
            "#9013FE",
            "#50E3C2",
            "#E8537A",
            "#B8E986",
            "#4A90D9",
            "#F8E71C",
        ]

        # --- Rows 2..n: per-channel absolute residuals ---
        for i, col in enumerate(channels):
            row_idx = i + 2
            color = channel_colors[i % len(channel_colors)]

            # Residual line
            fig.add_trace(
                go.Scatter(
                    x=df_pc_scores.index.tolist(),
                    y=df_pc_scores[col].tolist(),
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=1.5),
                    showlegend=True,
                    legendgroup=col,
                    hovertemplate=f"<b>{col}</b><br>Time: %{{x}}<br>|Residual|: %{{y:.4f}}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

            # Y-axis unit
            unit = self._get_channel_unit(col)
            if unit:
                fig.update_yaxes(title_text=f"|Residual| ({unit})", row=row_idx, col=1)

        # Layout
        fig.update_layout(
            height=max(600, 180 * n_rows),
            title={
                "text": (
                    f"Panel {panel_id}: Per-Channel Anomaly Scores"
                    "<br><sub>Each channel scored independently against its own quantile threshold</sub>"
                ),
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20, "color": COLORS["line"]},
            },
            hovermode="x unified",
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0, 0, 0, 0.5)",
                bordercolor=COLORS["border"],
                borderwidth=1,
            ),
        )

        # Update all axes styling
        for r in range(1, n_rows + 1):
            fig.update_xaxes(
                gridcolor="rgba(128, 128, 128, 0.15)",
                linecolor=COLORS["border"],
                row=r,
                col=1,
            )
            fig.update_yaxes(
                gridcolor="rgba(128, 128, 128, 0.15)",
                linecolor=COLORS["border"],
                row=r,
                col=1,
            )

        # X-axis label on the bottom only
        fig.update_xaxes(title_text="Timestamp", row=n_rows, col=1)

        return fig

    def _get_channel_unit(self, col_name: str) -> str:
        """Return the measurement unit label for a given channel column name."""
        col_lower = col_name.lower()
        if col_lower.startswith("exogenous_"):
            return "m3/h"
        if "durchfluss" in col_lower:
            return "l/h"
        if "temperature" in col_lower:
            return "\u00b0K"
        if "orp_mv" in col_lower:
            return "mV"
        if col_lower == "ph" or col_lower.endswith("_ph"):
            return "pH"
        if "conductivity" in col_lower:
            return "mS/cm"
        if "turbidity" in col_lower:
            return "NTU"
        if col_lower == "sac" or col_lower.endswith("_sac"):
            return "1/m"
        if "snow_depth" in col_lower:
            return "cm"
        return ""

    def _create_channels_overview_plot(
        self,
        panel_id: str,
        df_actual: pd.DataFrame,
        df_forecast: pd.DataFrame,
        df_flags: pd.DataFrame,
        data_age_seconds: float | None = None,
    ) -> go.Figure:
        """Create all channels overview plot.

        Args:
            panel_id: Panel identifier
            df_actual: DataFrame with actual values (timestamp as column)
            df_forecast: DataFrame with forecast values (timestamp as column)
            df_flags: DataFrame with anomaly flags
            data_age_seconds: If set, seconds since last valid sensor reading.
                When the gap is large a shaded "No Data" region is drawn.

        Returns:
            Plotly figure
        """
        # Get common columns (exclude 'timestamp' from channel columns)
        all_common_cols = df_actual.columns.intersection(df_forecast.columns)
        common_cols = [col for col in all_common_cols if col != "timestamp"]
        if len(common_cols) == 0:
            # Create empty figure if no common columns
            fig = go.Figure()
            fig.add_annotation(
                text="No matching columns between actual and forecast data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        # Ensure timestamp column exists
        if "timestamp" not in df_actual.columns or "timestamp" not in df_forecast.columns:
            self.logger.warning(f"Timestamp column missing, using index for panel {panel_id}")
            # Use index as timestamp
            timestamp_actual = (
                df_actual.index if isinstance(df_actual.index, pd.DatetimeIndex) else pd.to_datetime(df_actual.index)
            )
            (
                df_forecast.index
                if isinstance(df_forecast.index, pd.DatetimeIndex)
                else pd.to_datetime(df_forecast.index)
            )
        else:
            timestamp_actual = pd.to_datetime(df_actual["timestamp"])
            pd.to_datetime(df_forecast["timestamp"])

        n_channels = len(common_cols)
        n_cols = 2
        n_rows = (n_channels + 1) // 2

        # Create subplots
        subplot_titles = list(common_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # Get anomaly points
        # df_flags still has index, but we need to match with timestamp column in df_actual/df_forecast
        if "anomaly_flag" in df_flags.columns:
            anomaly_mask = df_flags["anomaly_flag"] == 1
            if anomaly_mask.any():
                # Get anomaly timestamps from flags index
                anomaly_times = df_flags.index[anomaly_mask]
            else:
                anomaly_times = pd.DatetimeIndex([])
        else:
            anomaly_times = pd.DatetimeIndex([])

        # Detect exogenous flow column for optional flow-weighted view.
        # Anomaly scoring multiplies residuals by this flow value, so we let
        # the user toggle between the raw values and the flow-weighted view
        # the scorer actually sees.
        flow_col = next((c for c in df_actual.columns if c.startswith("exogenous_")), None)
        flow_weighting_enabled = flow_col is not None and self.config.get("residual_weighting", {}).get(
            "enabled", False
        )
        if flow_weighting_enabled:
            flow_values_arr = df_actual[flow_col].to_numpy()
        else:
            flow_values_arr = None

        # Trace indices grouped by view, used to build the slider visibility.
        normal_trace_indices: list[int] = []
        flow_trace_indices: list[int] = []

        # Plot each channel
        for idx, col in enumerate(common_cols):
            row = idx // n_cols + 1
            col_num = idx % n_cols + 1

            # Get the column data - dataframes should already be aligned
            # Extract values and convert to lists to ensure proper JSON serialization
            # Plotly will serialize numpy arrays as binary, which causes issues in JavaScript
            actual_values = df_actual[col].values.tolist()
            forecast_values = df_forecast[col].values.tolist()

            # Use timestamp column if available (after reset_index), otherwise use index
            # Convert timestamps to lists as well for consistent serialization
            if "timestamp" in df_actual.columns:
                actual_times = pd.to_datetime(df_actual["timestamp"]).tolist()
            else:
                actual_times = df_actual.index.tolist()

            if "timestamp" in df_forecast.columns:
                forecast_times = pd.to_datetime(df_forecast["timestamp"]).tolist()
            else:
                forecast_times = df_forecast.index.tolist()

            # Sanity check: values should be sensor readings, not array indices
            # For flow data, values should be in reasonable range (e.g., 40-80), not 0-900
            if len(actual_values) > 10:
                min_val = min(actual_values)
                max_val = max(actual_values)
                if min_val == 0 and max_val > len(actual_values) * 0.9:
                    self.logger.error(
                        f"CRITICAL: Column {col} values look like indices! Min: {min_val}, Max: {max_val}, Length: {len(actual_values)}"
                    )
                    # This shouldn't happen, but if it does, log the first few values
                    self.logger.error(f"First 10 values: {actual_values[:10]}")

            # Build customdata so hovering on either trace shows both values
            # Each element is [forecast_value] for actual trace, [actual_value] for forecast trace
            actual_customdata = [[fv] for fv in forecast_values]
            forecast_customdata = [[av] for av in actual_values]

            # Actual - pass values and times explicitly
            fig.add_trace(
                go.Scatter(
                    x=actual_times,
                    y=actual_values,
                    customdata=actual_customdata,
                    mode="lines",
                    name="Actual",
                    line=dict(color=COLORS["line"], width=1.5),
                    showlegend=(idx == 0),
                    legendgroup="actual",
                    hovertemplate=(
                        f"<b>{col}</b><br>"
                        "Time: %{x}<br>"
                        "Actual: %{y:.4f}<br>"
                        "Forecast: %{customdata[0]:.4f}"
                        "<extra>Actual</extra>"
                    ),
                ),
                row=row,
                col=col_num,
            )
            normal_trace_indices.append(len(fig.data) - 1)

            # Forecast - pass values and times explicitly
            fig.add_trace(
                go.Scatter(
                    x=forecast_times,
                    y=forecast_values,
                    customdata=forecast_customdata,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="orange", width=1.5, dash="dash"),
                    showlegend=(idx == 0),
                    legendgroup="forecast",
                    hovertemplate=(
                        f"<b>{col}</b><br>"
                        "Time: %{x}<br>"
                        "Actual: %{customdata[0]:.4f}<br>"
                        "Forecast: %{y:.4f}"
                        "<extra>Forecast</extra>"
                    ),
                ),
                row=row,
                col=col_num,
            )
            normal_trace_indices.append(len(fig.data) - 1)

            # Flow-weighted actual & forecast (hidden until slider toggles).
            if flow_weighting_enabled and flow_values_arr is not None:
                actual_flow_values = (np.asarray(actual_values) * flow_values_arr).tolist()
                forecast_flow_values = (np.asarray(forecast_values) * flow_values_arr).tolist()
                actual_flow_customdata = [[fv] for fv in forecast_flow_values]
                forecast_flow_customdata = [[av] for av in actual_flow_values]

                fig.add_trace(
                    go.Scatter(
                        x=actual_times,
                        y=actual_flow_values,
                        customdata=actual_flow_customdata,
                        mode="lines",
                        name="Actual × flow",
                        line=dict(color=COLORS["line"], width=1.5),
                        showlegend=(idx == 0),
                        legendgroup="actual_flow",
                        visible=False,
                        hovertemplate=(
                            f"<b>{col}</b><br>"
                            "Time: %{x}<br>"
                            "Actual × flow: %{y:.4f}<br>"
                            "Forecast × flow: %{customdata[0]:.4f}"
                            "<extra>Actual × flow</extra>"
                        ),
                    ),
                    row=row,
                    col=col_num,
                )
                flow_trace_indices.append(len(fig.data) - 1)

                fig.add_trace(
                    go.Scatter(
                        x=forecast_times,
                        y=forecast_flow_values,
                        customdata=forecast_flow_customdata,
                        mode="lines",
                        name="Forecast × flow",
                        line=dict(color="orange", width=1.5, dash="dash"),
                        showlegend=(idx == 0),
                        legendgroup="forecast_flow",
                        visible=False,
                        hovertemplate=(
                            f"<b>{col}</b><br>"
                            "Time: %{x}<br>"
                            "Actual × flow: %{customdata[0]:.4f}<br>"
                            "Forecast × flow: %{y:.4f}"
                            "<extra>Forecast × flow</extra>"
                        ),
                    ),
                    row=row,
                    col=col_num,
                )
                flow_trace_indices.append(len(fig.data) - 1)

            # Anomalies
            if len(anomaly_times) > 0:
                # Get actual values at anomaly times
                if "timestamp" in df_actual.columns:
                    # Match by timestamp column
                    anomaly_mask = pd.to_datetime(df_actual["timestamp"]).isin(anomaly_times)
                    if anomaly_mask.any():
                        anomaly_values = df_actual.loc[anomaly_mask, col].values.tolist()
                        anomaly_times_plot = pd.to_datetime(df_actual.loc[anomaly_mask, "timestamp"]).tolist()
                    else:
                        anomaly_values = []
                        anomaly_times_plot = []
                else:
                    # Match by index
                    common_anomaly_times = df_actual.index.intersection(anomaly_times)
                    if len(common_anomaly_times) > 0:
                        anomaly_values = df_actual.loc[common_anomaly_times, col].values.tolist()
                        anomaly_times_plot = common_anomaly_times.tolist()
                    else:
                        anomaly_values = []
                        anomaly_times_plot = []

                if len(anomaly_values) > 0:
                    # Get forecast values at anomaly timestamps for hover comparison
                    anomaly_forecast_values = []
                    for at in anomaly_times_plot:
                        at_ts = pd.Timestamp(at)
                        if "timestamp" in df_forecast.columns:
                            fc_mask = pd.to_datetime(df_forecast["timestamp"]) == at_ts
                            if fc_mask.any():
                                anomaly_forecast_values.append(float(df_forecast.loc[fc_mask, col].iloc[0]))
                            else:
                                anomaly_forecast_values.append(None)
                        elif at_ts in df_forecast.index:
                            anomaly_forecast_values.append(float(df_forecast.loc[at_ts, col]))
                        else:
                            anomaly_forecast_values.append(None)

                    anomaly_customdata = [[fv] for fv in anomaly_forecast_values]
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_times_plot,
                            y=anomaly_values,
                            customdata=anomaly_customdata,
                            mode="markers",
                            name="Anomaly",
                            marker=dict(color=COLORS["anomaly_marker"], size=5, symbol="circle"),
                            showlegend=(idx == 0),
                            legendgroup="anomaly",
                            hovertemplate=(
                                f"<b>[! ] {col}</b><br>"
                                "Time: %{x}<br>"
                                "Actual: %{y:.4f}<br>"
                                "Forecast: %{customdata[0]:.4f}"
                                "<extra>Anomaly</extra>"
                            ),
                        ),
                        row=row,
                        col=col_num,
                    )
                    normal_trace_indices.append(len(fig.data) - 1)

                    if flow_weighting_enabled and flow_values_arr is not None:
                        # Lookup flow value per anomaly timestamp via the actual frame.
                        if "timestamp" in df_actual.columns:
                            flow_series = pd.Series(flow_values_arr, index=pd.to_datetime(df_actual["timestamp"]))
                        else:
                            flow_series = pd.Series(flow_values_arr, index=df_actual.index)
                        anomaly_flow = flow_series.reindex(pd.to_datetime(anomaly_times_plot)).to_numpy()
                        anomaly_flow_values = (np.asarray(anomaly_values) * anomaly_flow).tolist()
                        anomaly_flow_forecast = [
                            (fv * f) if (fv is not None and pd.notna(f)) else None
                            for fv, f in zip(anomaly_forecast_values, anomaly_flow.tolist(), strict=False)
                        ]
                        anomaly_flow_customdata = [[fv] for fv in anomaly_flow_forecast]
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_times_plot,
                                y=anomaly_flow_values,
                                customdata=anomaly_flow_customdata,
                                mode="markers",
                                name="Anomaly × flow",
                                marker=dict(color=COLORS["anomaly_marker"], size=5, symbol="circle"),
                                showlegend=(idx == 0),
                                legendgroup="anomaly_flow",
                                visible=False,
                                hovertemplate=(
                                    f"<b>[! ] {col}</b><br>"
                                    "Time: %{x}<br>"
                                    "Actual × flow: %{y:.4f}<br>"
                                    "Forecast × flow: %{customdata[0]:.4f}"
                                    "<extra>Anomaly × flow</extra>"
                                ),
                            ),
                            row=row,
                            col=col_num,
                        )
                        flow_trace_indices.append(len(fig.data) - 1)

            # Set y-axis unit label for this subplot.
            unit = self._get_channel_unit(col)
            if unit:
                fig.update_yaxes(title_text=unit, row=row, col=col_num)

            # Add hoverable source info button at the top-right of each channel subplot.
            axis_index = idx + 1
            axis_suffix = "" if axis_index == 1 else str(axis_index)
            source_label = self._get_channel_source_label(panel_id, col)
            fig.add_annotation(
                x=0.99,
                y=0.99,
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                text="i",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=12, color=COLORS["line"]),
                align="right",
                hovertext=f"Data source: {source_label}",
                captureevents=True,
            )

        # Draw a "No Data" shaded region when sensor data is stale
        if data_age_seconds is not None and data_age_seconds > 3600:
            data_end = timestamp_actual.max() if len(timestamp_actual) > 0 else None
            if data_end is not None:
                now_naive = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
                hours_missing = data_age_seconds / 3600
                gap_label = (
                    f"{int(hours_missing // 24)}d {int(hours_missing % 24)}h"
                    if hours_missing >= 24
                    else f"{int(hours_missing)}h"
                )
                for idx in range(n_channels):
                    row = idx // n_cols + 1
                    col_num = idx % n_cols + 1
                    fig.add_vrect(
                        x0=data_end,
                        x1=now_naive,
                        fillcolor="rgba(169, 15, 18, 0.25)",
                        line=dict(width=0),
                        layer="below",
                        row=row,
                        col=col_num,
                    )
                # Single annotation on the first subplot
                fig.add_annotation(
                    x=data_end + (now_naive - data_end) / 2,
                    y=0.5,
                    xref="x",
                    yref="y domain",
                    text=f"NO DATA ({gap_label})",
                    showarrow=False,
                    font=dict(size=14, color="#A90F12", family="monospace"),
                    bgcolor="rgba(0,0,0,0.6)",
                    bordercolor="#A90F12",
                    borderwidth=1,
                    borderpad=6,
                )

        # Build a button toggle that switches between raw values and
        # flow-weighted values (matching what the scorer actually sees).
        updatemenus = []
        if flow_weighting_enabled and flow_trace_indices:
            total_traces = len(fig.data)
            normal_visibility = [i in normal_trace_indices for i in range(total_traces)]
            flow_visibility = [i in flow_trace_indices for i in range(total_traces)]
            updatemenus = [
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.06,
                    yanchor="bottom",
                    pad=dict(l=0, r=0, t=0, b=0),
                    bgcolor="rgba(0, 0, 0, 0.4)",
                    bordercolor=COLORS["border"],
                    borderwidth=1,
                    font=dict(color="white", size=11),
                    buttons=[
                        dict(
                            method="update",
                            label="Actual",
                            args=[{"visible": normal_visibility}],
                        ),
                        dict(
                            method="update",
                            label=f"× {flow_col}",
                            args=[{"visible": flow_visibility}],
                        ),
                    ],
                )
            ]

        # Update layout
        fig.update_layout(
            height=max(600, 300 * n_rows),
            title_text=f"Panel {panel_id} - All Channels with Predictions",
            showlegend=True,
            hovermode="closest",
            hoverdistance=20,
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color="white"),
            legend=dict(bgcolor="rgba(0, 0, 0, 0.5)", bordercolor=COLORS["border"], borderwidth=1),
            updatemenus=updatemenus,
        )

        # Update axes with spike lines for precise timestamp targeting
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(
                    gridcolor="rgba(128, 128, 128, 0.15)",
                    linecolor=COLORS["border"],
                    showspikes=True,
                    spikecolor="rgba(255, 255, 255, 0.4)",
                    spikethickness=1,
                    spikedash="dot",
                    spikemode="across",
                    row=i,
                    col=j,
                )
                fig.update_yaxes(
                    gridcolor="rgba(128, 128, 128, 0.15)",
                    linecolor=COLORS["border"],
                    showspikes=True,
                    spikecolor="rgba(255, 255, 255, 0.4)",
                    spikethickness=1,
                    spikedash="dot",
                    spikemode="across",
                    row=i,
                    col=j,
                )

        # Update x-axis title on bottom row
        for j in range(1, n_cols + 1):
            fig.update_xaxes(title_text="Timestamp", row=n_rows, col=j)

        return fig

    def _assemble_html(
        self,
        figures: dict[str, go.Figure],
        event_contributions: dict[str, list[dict[str, Any]]],
        timestamp: datetime,
        output_path: Path,
        data_path: Path,
        metadata_path: Path,
        current_statuses: dict = None,
        fetch_status: dict[str, dict] | None = None,
    ) -> None:
        """Assemble HTML with live-updating Plotly charts.

        Args:
            figures: Dictionary mapping figure IDs to Plotly figures
            timestamp: Prediction timestamp
            output_path: Path to save HTML file
            data_path: Path to save figure JSON data
            metadata_path: Path to save metadata with timestamp
            current_statuses: Dictionary mapping panel_id to current status info
            fetch_status: Dictionary mapping source name to fetch status info
        """
        if fetch_status is None:
            fetch_status = {}
        if current_statuses is None:
            current_statuses = {}
        # Convert figures to JSON strings using Plotly's built-in serialization
        # This handles all numpy/pandas types automatically
        import json

        figure_jsons = {}
        for fig_id, fig in figures.items():
            # Use Plotly's to_json which handles all serialization properly
            figure_jsons[fig_id] = fig.to_json()

        # Save figure data to separate JSON file
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(figure_jsons, f)

        # Save metadata with timestamp
        metadata = {
            "last_update": timestamp.isoformat(),
            "last_update_unix": timestamp.timestamp(),
            "event_contributions": convert_numpy_to_list(event_contributions),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Format timestamp
        tz_label = timestamp.tzname() or self.timezone_name
        timestamp_str = timestamp.strftime(f"%Y-%m-%d %H:%M:%S {tz_label}")
        timestamp.isoformat()
        report_timezone_js = json.dumps(self.timezone_name)
        source_labels_js = json.dumps(
            {
                "primary": self.primary_name,
                "exogenous": self.exogenous_name,
                "weather": "Open-Meteo",
            }
        )

        # Build weather source/place label for header
        weather_cfg = self._exogenous_source_config("weather")
        weather_enabled = weather_cfg.get("enabled", True) and weather_cfg.get("latitude") is not None
        weather_lat = weather_cfg.get("latitude")
        weather_lon = weather_cfg.get("longitude")
        weather_place = weather_cfg.get("place") or weather_cfg.get("location")
        if weather_enabled and weather_lat is not None and weather_lon is not None:
            if weather_place:
                weather_source_label = f"Open-Meteo ({weather_place}, {weather_lat:.4f}, {weather_lon:.4f})"
            else:
                weather_source_label = f"Open-Meteo ({weather_lat:.4f}, {weather_lon:.4f})"
        elif weather_enabled:
            weather_source_label = "Open-Meteo (location not configured)"
        else:
            weather_source_label = "Weather data disabled"

        # Generate HTML - use regular string concatenation to avoid f-string nesting issues
        # Add meta refresh for file:// protocol (page reload every 30 seconds)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Anomaly Detection Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <!-- Note: Tailwind CDN is used here for standalone HTML file portability -->
    <!-- For production, consider using Tailwind CLI or PostCSS plugin -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/flowbite@2.4.1/dist/flowbite.min.css" rel="stylesheet" />
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        'custom-bg': '#090909',
                        'custom-border': '#A90F12',
                        'custom-line': '#5EA6E1',
                    }}
                }}
            }}
        }}
    </script>
</head>
<body class="bg-custom-bg text-white min-h-screen">
    <!-- Header with metadata -->
    <header class="bg-gray-900 border-b-2 border-custom-border p-6">
        <div class="max-w-7xl mx-auto flex items-start justify-between gap-6">
            <!-- Title + metadata -->
            <div>
                <h1 class="text-3xl font-bold text-custom-line mb-4">Live Anomaly Detection Report</h1>
                <div class="flex gap-8 text-sm text-gray-300">
                    <div class="flex items-center gap-2">
                        <span class="font-semibold">Last Update:</span>
                        <span id="timestamp" class="font-mono text-custom-line">{timestamp_str}</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="font-semibold">Time Elapsed:</span>
                        <span id="elapsed" class="font-mono text-green-400">00:00:00</span>
                    </div>
                </div>
                <div class="mt-2 text-sm text-gray-300">
                    <span class="font-semibold">Weather Data Source:</span>
                    <span id="weather-source" class="font-mono text-custom-line">{weather_source_label}</span>
                </div>
            </div>
            <!-- Logo top-right -->
            <div class="bg-white rounded-lg px-3 py-2 flex-shrink-0 shadow-md">
                <img src="./spotlogo_red.png" alt="Spotanomaly" style="height:48px;width:auto;display:block;">
            </div>
        </div>
        </header>

    <!-- Stale data warning banner (hidden by default, shown by JS) -->
    <div id="stale-warning" class="hidden bg-yellow-900/80 border-2 border-yellow-500 text-yellow-200 px-6 py-4 text-center">
        <span class="text-xl font-bold">Warning: Data may be stale</span>
        <p id="stale-warning-detail" class="text-sm mt-1">The last successful update was more than expected. Check data sources below.</p>
    </div>

    <!-- Main content -->
    <main class="max-w-7xl mx-auto p-6 space-y-12">
        <!-- Data Source Status -->
        <section id="data-source-status" class="space-y-4">
            <h2 class="text-2xl font-bold border-l-4 border-custom-border pl-4">Data Source Status</h2>
            <div id="data-source-cards" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <!-- Data source cards will be inserted here by JavaScript -->
            </div>
        </section>

        <!-- Current Status Overview -->
        <section id="status-overview" class="space-y-4">
            <h2 class="text-2xl font-bold border-l-4 border-custom-border pl-4">Current Status</h2>
            <div id="status-cards" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <!-- Status cards will be inserted here by JavaScript -->
            </div>
        </section>
"""

        # Extract panel IDs from figure keys (close f-string, use regular strings)
        panel_ids = set()
        for fig_id in figure_jsons.keys():
            if fig_id.startswith("panel") and "_" in fig_id:
                panel_id = fig_id.split("_")[0].replace("panel", "")
                panel_ids.add(panel_id)

        # Sort panel IDs for consistent ordering
        panel_ids = sorted(panel_ids)

        # Add panels
        for panel_id in panel_ids:
            panel_key_prefix = f"panel{panel_id}_"
            panel_figures = {k: v for k, v in figure_jsons.items() if k.startswith(panel_key_prefix)}

            if not panel_figures:
                continue

            # Per-panel stale-data warning
            panel_gap = (
                (fetch_status.get("primary", {}).get("panel_nan_gaps", {}).get(str(panel_id), {}))
                if fetch_status
                else {}
            )
            panel_age_sec = panel_gap.get("data_age_seconds") or 0
            panel_stale_html = ""
            if panel_age_sec > 3600:
                hours = panel_age_sec / 3600
                if hours >= 24:
                    age_str = f"{int(hours // 24)}d {int(hours % 24)}h"
                else:
                    age_str = f"{int(hours)}h {int((hours % 1) * 60)}m"
                last_valid = panel_gap.get("last_valid_timestamp", "unknown")
                panel_stale_html = (
                    f'                <div class="bg-red-900/30 border-2 border-red-500 '
                    f'rounded-lg p-4 flex items-start gap-3">\n'
                    f'                    <span class="text-2xl leading-none">&#x26A0;</span>\n'
                    f"                    <div>\n"
                    f'                        <p class="text-red-300 font-bold">'
                    f"Sensor data is {age_str} old</p>\n"
                    f'                        <p class="text-sm text-red-400/80">'
                    f"Last valid reading: {last_valid}. "
                    f"The charts below show the most recent available data, "
                    f"not the current state.</p>\n"
                    f"                    </div>\n"
                    f"                </div>\n"
                )

            html_content += f"""        <!-- Panel {panel_id} Section -->
        <section class="space-y-6">
            <h2 class="text-2xl font-bold border-l-4 border-custom-border pl-4">Panel {panel_id}</h2>
            <div class="space-y-8">
{panel_stale_html}"""

            # Add anomaly scores section with toggle between combined and per-channel
            has_combined = f"{panel_key_prefix}normalized" in panel_figures
            has_per_channel = f"{panel_key_prefix}per_channel" in panel_figures

            if has_combined or has_per_channel:
                # Toggle buttons (only shown when both views are available)
                toggle_html = ""
                if has_combined and has_per_channel:
                    toggle_html = f"""
                    <div class="flex gap-2 mb-3">
                        <button id="btn_{panel_key_prefix}combined" onclick="toggleScoringView('{panel_id}', 'combined')" class="px-4 py-1.5 text-sm font-semibold rounded border-2 border-custom-line bg-custom-line/20 text-custom-line">Combined Score</button>
                        <button id="btn_{panel_key_prefix}per_channel" onclick="toggleScoringView('{panel_id}', 'per_channel')" class="px-4 py-1.5 text-sm font-semibold rounded border-2 border-gray-600 text-gray-400 hover:border-gray-400">Per-Channel Scores</button>
                    </div>"""

                html_content += f"""                <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-3 text-custom-line">Anomaly Scores</h3>{toggle_html}"""

                if has_combined:
                    html_content += f"""
                    <div id="view_{panel_key_prefix}combined">
                        <div id="{panel_key_prefix}normalized"></div>
                    </div>"""

                if has_per_channel:
                    # Hidden by default if combined is also available
                    display_style = ' style="display:none;"' if has_combined else ""
                    html_content += f"""
                    <div id="view_{panel_key_prefix}per_channel"{display_style}>
                        <div id="{panel_key_prefix}per_channel"></div>
                    </div>"""

                html_content += """
                </div>
"""

            # Add per-event contributions plot
            if f"{panel_key_prefix}event_contributions" in panel_figures:
                num_events = len(event_contributions.get(panel_id, []))
                # ~3 * per-row-height + title, matching sizing in
                # _create_event_contributions_plot so ~3 events fit before scroll.
                if num_events > 3:
                    plot_wrapper_open = '<div class="overflow-y-auto" style="max-height: 740px;">'
                    plot_wrapper_close = "</div>"
                    scroll_hint = (
                        '                    <p class="text-xs text-gray-400 mb-2">'
                        f"Showing {num_events} events — scroll inside the panel to see all.</p>\n"
                    )
                else:
                    plot_wrapper_open = ""
                    plot_wrapper_close = ""
                    scroll_hint = ""
                html_content += f"""                <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-3 text-custom-line">Per-Event Contributions (Grouped)</h3>
{scroll_hint}                    {plot_wrapper_open}<div id="{panel_key_prefix}event_contributions"></div>{plot_wrapper_close}
                </div>
"""
            else:
                html_content += """                <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
                    <div class="text-center py-12">
                        <p class="text-lg text-custom-line font-semibold">[OK] No anomalies detected</p>
                        <p class="text-sm text-gray-400 mt-2">This panel appears to be operating normally during the analysis period.</p>
                    </div>
                </div>
"""

            # Add channels overview plot
            if f"{panel_key_prefix}channels" in panel_figures:
                html_content += f"""                <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-3 text-custom-line">All Channels Overview</h3>
                    <div id="{panel_key_prefix}channels"></div>
                </div>
"""

            html_content += """            </div>
        </section>

"""

        html_content += """    </main>

    <!-- Embedded data for offline/file:// protocol use -->
    <script id="embedded-status" type="application/json">
"""
        # Embed current status data
        import json

        status_json = json.dumps(current_statuses, indent=4, ensure_ascii=False)
        html_content += status_json

        html_content += """
    </script>

    <script id="embedded-figures" type="application/json">
"""

        # Embed the figure data as JSON for file:// protocol
        import json

        figures_dict = {}
        for fig_id, fig_json_str in figure_jsons.items():
            try:
                figures_dict[fig_id] = json.loads(fig_json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON for {fig_id}: {e}")
                continue

        embedded_json = json.dumps(figures_dict, indent=4, ensure_ascii=False)
        html_content += embedded_json

        html_content += """
    </script>

    <script id="embedded-metadata" type="application/json">
"""
        # Add metadata JSON
        html_content += json.dumps(metadata, ensure_ascii=False)

        html_content += """
    </script>

    <script id="embedded-fetch-status" type="application/json">
"""
        html_content += json.dumps(fetch_status, indent=4, ensure_ascii=False)

        html_content += r"""
    </script>

    <script>
        // Global state
        let predictionTime = null;
        let reportMetadata = null;
        const pinnedEventByPanel = {};
        let eventSource = null;
        let isFileProtocol = window.location.protocol === 'file:';
        let refreshInterval = null;
        const reportTimezone = {report_timezone_js};

        function formatDateInReportTimezone(value) {
            if (!value) return 'N/A';

            // If backend already sent a timezone-naive timestamp (report timezone wall-clock),
            // show it directly to avoid double timezone conversion in the browser.
            if (typeof value === 'string') {
                const hasOffset = /Z$|[+-]\d{2}:\d{2}$/i.test(value);
                if (!hasOffset) {
                    return value.replace('T', ' ');
                }
            }

            const dateValue = new Date(value);
            if (Number.isNaN(dateValue.getTime())) return 'N/A';
            const options = {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                timeZoneName: 'short'
            };
            if (reportTimezone && reportTimezone.toLowerCase() !== 'local') {
                options.timeZone = reportTimezone;
            }
            return dateValue.toLocaleString('en-US', options);
        }

        // Toggle between combined and per-channel scoring views
        function toggleScoringView(panelId, view) {
            const prefix = 'panel' + panelId + '_';
            const combinedView = document.getElementById('view_' + prefix + 'combined');
            const perChannelView = document.getElementById('view_' + prefix + 'per_channel');
            const btnCombined = document.getElementById('btn_' + prefix + 'combined');
            const btnPerChannel = document.getElementById('btn_' + prefix + 'per_channel');

            const activeClasses = 'border-custom-line bg-custom-line/20 text-custom-line';
            const inactiveClasses = 'border-gray-600 text-gray-400 hover:border-gray-400';

            if (view === 'combined') {
                if (combinedView) combinedView.style.display = '';
                if (perChannelView) perChannelView.style.display = 'none';
                if (btnCombined) { btnCombined.className = 'px-4 py-1.5 text-sm font-semibold rounded border-2 ' + activeClasses; }
                if (btnPerChannel) { btnPerChannel.className = 'px-4 py-1.5 text-sm font-semibold rounded border-2 ' + inactiveClasses; }
            } else {
                if (combinedView) combinedView.style.display = 'none';
                if (perChannelView) perChannelView.style.display = '';
                if (btnCombined) { btnCombined.className = 'px-4 py-1.5 text-sm font-semibold rounded border-2 ' + inactiveClasses; }
                if (btnPerChannel) { btnPerChannel.className = 'px-4 py-1.5 text-sm font-semibold rounded border-2 ' + activeClasses; }
                // Trigger Plotly relayout so chart renders correctly after being unhidden
                const plotEl = document.getElementById(prefix + 'per_channel');
                if (plotEl && plotEl.data) {
                    Plotly.Plots.resize(plotEl);
                }
            }
        }

        // Show info message
        if (isFileProtocol) {
            console.log('[FILE] Opened directly - showing embedded data snapshot');
            console.log('[AUTO] Page will auto-refresh every 30 seconds');

            console.log('💡 For live auto-updates, run: spotanomaly2 live --interval 5');
            console.log('🌐 Then open: http://localhost:{live_report_server_port} (or this host IP from another device)');
        } else {
            console.log('[HTTP] Live updates enabled');
        }

        // Load embedded data (for file:// protocol)
        function loadEmbeddedData() {
            const figuresElement = document.getElementById('embedded-figures');
            const metadataElement = document.getElementById('embedded-metadata');
            const statusElement = document.getElementById('embedded-status');
            const fetchStatusElement = document.getElementById('embedded-fetch-status');

            if (figuresElement && metadataElement) {
                return {
                    figures: JSON.parse(figuresElement.textContent),
                    metadata: JSON.parse(metadataElement.textContent),
                    status: statusElement ? JSON.parse(statusElement.textContent) : null,
                    fetchStatus: fetchStatusElement ? JSON.parse(fetchStatusElement.textContent) : null
                };
            }
            return null;
        }

        function parseTimestamp(value) {
            if (!value) return null;
            const ts = new Date(value);
            return Number.isNaN(ts.getTime()) ? null : ts;
        }

        function getEventIndexForTimestamp(panelId, pointTimestamp) {
            if (!reportMetadata || !reportMetadata.event_contributions) return -1;

            const events = reportMetadata.event_contributions[String(panelId)] || [];
            if (!events.length) return -1;

            const pointTs = parseTimestamp(pointTimestamp);
            if (!pointTs) return -1;

            for (let i = 0; i < events.length; i += 1) {
                const event = events[i];
                const startTs = parseTimestamp(event.region_start);
                const endTs = parseTimestamp(event.region_end);

                if (startTs && endTs && pointTs >= startTs && pointTs <= endTs) {
                    return i;
                }

                const eventTs = parseTimestamp(event.timestamp);
                if (eventTs && Math.abs(pointTs - eventTs) <= 60 * 1000) {
                    return i;
                }
            }

            return -1;
        }

        function getEventIndexFromPoint(panelId, point) {
            if (!point) return -1;

            // Prefer timestamp lookup so mapping stays correct when region order changes.
            const timestampIdx = getEventIndexForTimestamp(panelId, point.x);
            if (timestampIdx >= 0) return timestampIdx;

            const customdata = point.customdata;
            if (Array.isArray(customdata) && customdata.length > 0) {
                const idx = Number(customdata[0]);
                if (Number.isInteger(idx) && idx >= 0) return idx;
            } else if (customdata !== undefined && customdata !== null) {
                const idx = Number(customdata);
                if (Number.isInteger(idx) && idx >= 0) return idx;
            }

            return -1;
        }

        function highlightEventContribution(panelId, eventIndex) {
            const contributionPlotId = 'panel' + panelId + '_event_contributions';
            const contributionElement = document.getElementById(contributionPlotId);
            if (!contributionElement || !contributionElement.data || eventIndex < 0) return;

            const traceCount = contributionElement.data.length;
            const opacities = [];
            for (let i = 0; i < traceCount; i += 1) {
                opacities.push(i === eventIndex ? 1.0 : 0.2);
            }

            Plotly.restyle(contributionElement, {'opacity': opacities});
        }

        function resetEventContributionHighlight(panelId) {
            if (pinnedEventByPanel[panelId] !== undefined && pinnedEventByPanel[panelId] !== null) {
                highlightEventContribution(panelId, pinnedEventByPanel[panelId]);
                return;
            }

            const contributionPlotId = 'panel' + panelId + '_event_contributions';
            const contributionElement = document.getElementById(contributionPlotId);
            if (!contributionElement || !contributionElement.data) return;

            const traceCount = contributionElement.data.length;
            const opacities = [];
            for (let i = 0; i < traceCount; i += 1) {
                opacities.push(1.0);
            }

            Plotly.restyle(contributionElement, {'opacity': opacities});
        }

        function clearPinnedEvent(panelId) {
            delete pinnedEventByPanel[panelId];
            resetEventContributionHighlight(panelId);
        }

        function clearAllPinnedEvents() {
            const panelIds = Object.keys(pinnedEventByPanel);
            for (let i = 0; i < panelIds.length; i += 1) {
                const panelId = panelIds[i];
                delete pinnedEventByPanel[panelId];
                resetEventContributionHighlight(panelId);
            }
        }

        function setupAnomalyContributionLinking(figId, element) {
            const panelMatch = figId.match(/^panel(.+)_normalized$/);
            if (!panelMatch) return;
            const panelId = panelMatch[1];

            if (typeof element.removeAllListeners === 'function') {
                element.removeAllListeners('plotly_hover');
                element.removeAllListeners('plotly_click');
                element.removeAllListeners('plotly_unhover');
            }

            element.on('plotly_hover', function(evt) {
                if (!evt || !evt.points || evt.points.length === 0) return;
                for (let i = 0; i < evt.points.length; i += 1) {
                    const point = evt.points[i];
                    if (!point) continue;
                    const eventIndex = getEventIndexFromPoint(panelId, point);
                    if (eventIndex >= 0) {
                        highlightEventContribution(panelId, eventIndex);
                        return;
                    }
                }

                resetEventContributionHighlight(panelId);
            });

            element.on('plotly_click', function(evt) {
                if (!evt || !evt.points || evt.points.length === 0) return;
                let eventIndex = -1;
                for (let i = 0; i < evt.points.length; i += 1) {
                    const point = evt.points[i];
                    if (!point) continue;
                    eventIndex = getEventIndexFromPoint(panelId, point);
                    if (eventIndex >= 0) {
                        break;
                    }
                }

                if (eventIndex < 0) {
                    clearPinnedEvent(panelId);
                    return;
                }

                if (pinnedEventByPanel[panelId] === eventIndex) {
                    clearPinnedEvent(panelId);
                    return;
                }

                pinnedEventByPanel[panelId] = eventIndex;
                highlightEventContribution(panelId, eventIndex);

                const contributionPlotId = 'panel' + panelId + '_event_contributions';
                const contributionElement = document.getElementById(contributionPlotId);
                if (contributionElement) {
                    contributionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            });

            element.on('plotly_unhover', function() {
                resetEventContributionHighlight(panelId);
            });
        }

        // Global keyboard shortcut: Esc clears all pinned contribution focus.
        document.addEventListener('keydown', function(evt) {
            if (evt.key === 'Escape') {
                clearAllPinnedEvents();
            }
        });

        // Data source status rendering
        const SOURCE_LABELS = {source_labels_js};
        const SOURCE_ORDER = ['primary', 'exogenous', 'weather'];

        function renderDataSourceStatus(fetchStatus) {
            const container = document.getElementById('data-source-cards');
            if (!container) return;
            container.innerHTML = '';

            if (!fetchStatus || Object.keys(fetchStatus).length === 0) {
                container.innerHTML = '<p class="text-gray-500 text-sm col-span-3">No data source status available.</p>';
                return;
            }

            SOURCE_ORDER.forEach(function(key) {
                const info = fetchStatus[key];
                if (!info) return;

                const label = SOURCE_LABELS[key] || key;
                const status = info.status || 'unknown';
                const error = info.error || null;
                const ts = info.timestamp ? formatDateInReportTimezone(info.timestamp) : 'N/A';

                let statusColor, statusIcon, statusText, borderColor;
                switch (status) {
                    case 'ok':
                        statusColor = 'text-green-400';
                        borderColor = 'border-green-500 bg-green-900/20';
                        statusIcon = 'o';
                        statusText = 'OK';
                        break;
                    case 'degraded':
                        statusColor = 'text-yellow-400';
                        borderColor = 'border-yellow-500 bg-yellow-900/20';
                        statusIcon = 'o';
                        statusText = 'Degraded';
                        break;
                    case 'error':
                        statusColor = 'text-red-400';
                        borderColor = 'border-red-500 bg-red-900/20';
                        statusIcon = 'o';
                        statusText = 'Error';
                        break;
                    case 'disabled':
                        statusColor = 'text-gray-500';
                        borderColor = 'border-gray-600 bg-gray-800/30';
                        statusIcon = '-';
                        statusText = 'Disabled';
                        break;
                    default:
                        statusColor = 'text-gray-500';
                        borderColor = 'border-gray-600 bg-gray-800/30';
                        statusIcon = '?';
                        statusText = 'Unknown';
                }

                const card = document.createElement('div');
                card.className = 'bg-gray-900 rounded-lg p-4 border-2 ' + borderColor;

                let errorHtml = '';
                if (error) {
                    const safeError = error.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    errorHtml = '<p class="text-xs text-red-300 mt-2 font-mono break-all">' + safeError + '</p>';
                }

                // Per-panel data freshness details (primary only)
                let nanGapHtml = '';
                const panelGaps = info.panel_nan_gaps || null;
                if (panelGaps && Object.keys(panelGaps).length > 0) {
                    let gapRows = '';
                    Object.keys(panelGaps).sort().forEach(function(pid) {
                        const gap = panelGaps[pid];
                        const nanRows = gap.trailing_nan_rows || 0;
                        const ageSec = gap.data_age_seconds || 0;
                        const lastValid = gap.last_valid_timestamp
                            ? formatDateInReportTimezone(gap.last_valid_timestamp) : 'never';

                        let ageLabel = '';
                        let rowColor = 'text-green-400';
                        let statusLabel = 'OK';

                        if (ageSec > 86400) {
                            const days = Math.floor(ageSec / 86400);
                            const hrs = Math.floor((ageSec % 86400) / 3600);
                            ageLabel = days + 'd ' + hrs + 'h old';
                            rowColor = 'text-red-400';
                            statusLabel = ageLabel;
                        } else if (ageSec > 3600) {
                            const hrs = Math.floor(ageSec / 3600);
                            const mins = Math.floor((ageSec % 3600) / 60);
                            ageLabel = hrs + 'h ' + mins + 'm old';
                            rowColor = 'text-yellow-400';
                            statusLabel = ageLabel;
                        } else if (nanRows > 0) {
                            const dur = gap.trailing_nan_duration || (nanRows + ' rows');
                            rowColor = nanRows >= 12 ? 'text-red-400' : 'text-yellow-400';
                            statusLabel = dur + ' of NaN';
                        }

                        gapRows += '<tr><td class="pr-3 ' + rowColor + '">Panel ' + pid + '</td>' +
                            '<td class="text-right ' + rowColor + '">' + statusLabel + '</td></tr>';
                        if (statusLabel !== 'OK') {
                            gapRows += '<tr><td colspan="2" class="text-gray-500 text-[10px] pb-1">Last valid: ' + lastValid + '</td></tr>';
                        }
                    });
                    nanGapHtml = '<div class="mt-3 pt-3 border-t border-gray-700">' +
                        '<p class="text-xs font-semibold text-gray-400 mb-1">Sensor Data Freshness</p>' +
                        '<table class="w-full text-xs font-mono">' + gapRows + '</table></div>';
                }

                card.innerHTML = '<div class="flex items-center justify-between mb-2">' +
                    '<h3 class="text-lg font-bold text-white">' + label + '</h3>' +
                    '<span class="text-2xl ' + statusColor + '">' + statusIcon + '</span>' +
                    '</div>' +
                    '<div class="flex items-center justify-between">' +
                    '<span class="text-sm font-semibold ' + statusColor + '">' + statusText + '</span>' +
                    '<span class="text-xs text-gray-400 font-mono">' + ts + '</span>' +
                    '</div>' +
                    errorHtml +
                    nanGapHtml;

                container.appendChild(card);
            });
        }

        function checkStaleData() {
            const warningEl = document.getElementById('stale-warning');
            const detailEl = document.getElementById('stale-warning-detail');
            if (!warningEl) return;

            const reasons = [];

            // Check pipeline staleness
            if (predictionTime) {
                const now = new Date();
                const elapsedMs = now - predictionTime;
                const thresholdMs = 5 * 60 * 1000;
                if (elapsedMs > thresholdMs) {
                    const mins = Math.floor(elapsedMs / 60000);
                    reasons.push('Last pipeline update was ' + mins + ' minute(s) ago');
                }
            }

            // Check for stale sensor data or large NaN gaps
            const embedded = loadEmbeddedData();
            const fs = (embedded && embedded.fetchStatus) ? embedded.fetchStatus : null;
            if (fs && fs.primary && fs.primary.panel_nan_gaps) {
                const gaps = fs.primary.panel_nan_gaps;
                Object.keys(gaps).forEach(function(pid) {
                    const g = gaps[pid];
                    const ageSec = g.data_age_seconds || 0;
                    if (ageSec > 3600) {
                        const hrs = Math.floor(ageSec / 3600);
                        reasons.push('Panel ' + pid + ': sensor data is ' + hrs + 'h old');
                    } else if (g.trailing_nan_rows >= 12) {
                        const dur = g.trailing_nan_duration || (g.trailing_nan_rows + ' rows');
                        reasons.push('Panel ' + pid + ': sensor data is NaN for ' + dur);
                    }
                });
            }

            if (reasons.length > 0) {
                warningEl.classList.remove('hidden');
                if (detailEl) {
                    detailEl.textContent = reasons.join(' | ');
                }
            } else {
                warningEl.classList.add('hidden');
            }
        }

        // Render status cards with last 10 datapoints timeline
        function renderStatusCards(statusData) {
            if (!statusData) return;

            const container = document.getElementById('status-cards');
            if (!container) return;

            container.innerHTML = '';

            // Look up per-panel data age from fetch status
            const embedded = loadEmbeddedData();
            const panelGaps = (embedded && embedded.fetchStatus
                && embedded.fetchStatus.primary
                && embedded.fetchStatus.primary.panel_nan_gaps)
                ? embedded.fetchStatus.primary.panel_nan_gaps : {};

            Object.keys(statusData).forEach(function(panelId) {
                const status = statusData[panelId];
                const hasAnomaly = status.has_anomaly;
                const timestamp = status.timestamp ? formatDateInReportTimezone(status.timestamp) : 'N/A';
                const score = (status.score_normalized !== null && status.score_normalized !== undefined)
                    ? (status.score_normalized * 100).toFixed(1) + '%' : (status.score !== null ? status.score.toFixed(2) : 'N/A');

                // Check if data for this panel is stale
                const gap = panelGaps[String(panelId)] || {};
                const ageSec = gap.data_age_seconds || 0;
                const isStale = ageSec > 3600;

                let statusColor, statusIcon, statusText, textColor;
                if (isStale) {
                    const hrs = Math.floor(ageSec / 3600);
                    const ageLabel = ageSec > 86400
                        ? Math.floor(ageSec / 86400) + 'd ' + Math.floor((ageSec % 86400) / 3600) + 'h'
                        : hrs + 'h';
                    statusColor = 'border-red-500 bg-red-900/20';
                    statusIcon = '!';
                    statusText = 'STALE DATA (' + ageLabel + ' old)';
                    textColor = 'text-red-400';
                } else if (hasAnomaly) {
                    statusColor = 'border-red-500 bg-red-900/20';
                    statusIcon = '!';
                    statusText = 'ANOMALY DETECTED';
                    textColor = 'text-red-400';
                } else {
                    statusColor = 'border-green-500 bg-green-900/20';
                    statusIcon = 'OK';
                    statusText = 'Normal';
                    textColor = 'text-green-400';
                }

                // Build timeline of last 12 points
                const timelineLabel = isStale
                    ? 'Last 12 Data Points (stale - not current):'
                    : 'Last 12 Data Points (1 hour):';
                let timelineHtml = '<div class="mt-3 pt-3 border-t border-gray-700"><p class="text-xs font-semibold text-gray-400 mb-2">' + timelineLabel + '</p><div class="flex gap-1">';
                const recentPoints = status.recent_points || [];

                // Pad with empty rectangles if less than 12 points
                const numPoints = 12;
                const emptyPoints = numPoints - recentPoints.length;

                for (let i = 0; i < emptyPoints; i++) {
                    timelineHtml += '<div class="flex-1 h-8 bg-gray-700 rounded border border-gray-600" title="No data"></div>';
                }

                recentPoints.forEach(function(point, idx) {
                    const pointAnomaly = point.has_anomaly;
                    let bgColor, borderColor;
                    if (isStale) {
                        bgColor = 'bg-gray-600';
                        borderColor = 'border-gray-500';
                    } else if (pointAnomaly) {
                        bgColor = 'bg-red-500';
                        borderColor = 'border-red-400';
                    } else {
                        bgColor = 'bg-green-500';
                        borderColor = 'border-green-400';
                    }
                    const pointTime = point.timestamp
                        ? formatDateInReportTimezone(point.timestamp)
                        : 'N/A';
                    const pointScore = (point.score_normalized !== null && point.score_normalized !== undefined)
                        ? (point.score_normalized * 100).toFixed(1) + '%' : (point.score !== null ? point.score.toFixed(2) : 'N/A');
                    const tooltipText = pointTime;
                    const tooltipTextEscaped = tooltipText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                    const tooltipId = 'tooltip-panel' + panelId + '-point-' + idx;

                    timelineHtml += '<div class="relative flex-1">' +
                        '<div data-tooltip-target="' + tooltipId + '" data-tooltip-placement="top" class="h-8 flex items-center justify-center ' + bgColor + ' rounded border-2 ' + borderColor + ' cursor-help text-white text-[10px] font-medium">' + pointScore + '</div>' +
                        '<div id="' + tooltipId + '" role="tooltip" class="absolute z-50 invisible inline-block px-2 py-1 text-xs font-medium text-white transition-opacity duration-300 bg-gray-900 rounded shadow-sm opacity-0 tooltip">' +
                        tooltipTextEscaped +
                        '<div class="tooltip-arrow" data-popper-arrow></div></div></div>';
                });

                timelineHtml += '</div><p class="text-xs text-gray-500 mt-1 flex justify-between"><span>oldest</span><span>newest</span></p></div>';

                // Build channel list if anomaly
                let channelsHtml = '';
                if (hasAnomaly && status.anomaly_channels && status.anomaly_channels.length > 0) {
                    channelsHtml = '<div class="mt-3 pt-3 border-t border-gray-700"><p class="text-xs font-semibold text-gray-400 mb-2">Affected Channels:</p><ul class="text-xs text-gray-300 space-y-1">';
                    status.anomaly_channels.forEach(function(channel) {
                        const value = status.all_values[channel];
                        const valueStr = value !== null && value !== undefined ? value.toFixed(4) : 'N/A';
                        channelsHtml += '<li class="flex justify-between"><span class="font-mono text-red-300">* ' + channel + '</span><span class="text-gray-400">' + valueStr + '</span></li>';
                    });
                    channelsHtml += '</ul></div>';
                } else if (hasAnomaly) {
                    channelsHtml = '<div class="mt-3 pt-3 border-t border-gray-700"><p class="text-xs text-gray-400">All channels potentially affected</p></div>';
                }

                const card = document.createElement('div');
                card.className = 'bg-gray-900 rounded-lg p-4 border-2 ' + statusColor;
                card.innerHTML = `
                    <div class="flex items-start justify-between mb-3">
                        <div>
                            <h3 class="text-lg font-bold text-white">Panel ${panelId}</h3>
                            <p class="text-xs text-gray-400 font-mono mt-1">${timestamp}</p>
                        </div>
                        <div class="text-3xl">${statusIcon}</div>
                    </div>
                    <div class="space-y-2">
                        <div class="flex items-center justify-between">
                            <span class="text-sm font-semibold ${textColor}">${statusText}</span>
                            <span class="text-sm text-gray-400">Score: <span class="font-mono ${hasAnomaly ? 'text-red-300' : 'text-gray-300'}">${score}</span></span>
                        </div>
                        ${timelineHtml}
                        ${channelsHtml}
                    </div>
                `;
                container.appendChild(card);
            });
            if (typeof window.initFlowbite === 'function') {
                window.initFlowbite();
            }
        }

        // Render figures from data object
        function renderFigures(figuresData) {
            if (!figuresData) return false;

            Object.keys(figuresData).forEach(function(figId) {
                const element = document.getElementById(figId);
                if (element) {
                    try {
                        let figData = figuresData[figId];

                        // Parse if it's a JSON string (from server), otherwise use as-is (embedded)
                        if (typeof figData === 'string') {
                            figData = JSON.parse(figData);
                        }

                        // Verify data structure before plotting
                        if (!figData || !figData.data || !Array.isArray(figData.data)) {
                            console.error('Invalid figure data for ' + figId);
                            return;
                        }

                        const plotConfig = {
                            displayModeBar: true,
                            displaylogo: false,
                            responsive: true
                        };
                        // Event-contribution plots have a variable number of
                        // subplots (one per detected event). Plotly.react does not
                        // reliably update when the subplot structure changes, so
                        // for those we purge and fully re-plot instead.
                        if (figId.endsWith('_event_contributions')) {
                            Plotly.purge(element);
                            Plotly.newPlot(element, figData.data, figData.layout, plotConfig);
                        } else {
                            Plotly.react(element, figData.data, figData.layout, plotConfig);
                        }

                        if (figId.endsWith('_normalized')) {
                            setupAnomalyContributionLinking(figId, element);
                        }
                    } catch (e) {
                        console.error('Error rendering plot ' + figId + ':', e);
                    }
                }
            });

            return true;
        }

        // Load and render status - always try embedded first, then server
        async function loadAndRenderStatus(forceServer) {
            if (!forceServer && isFileProtocol) {
                const embedded = loadEmbeddedData();
                if (embedded && embedded.status) {
                    renderStatusCards(embedded.status);
                    return;
                }
            }

            if (isFileProtocol) return;

            try {
                const response = await fetch('current_status.json');
                if (!response.ok) return;
                const statusData = await response.json();
                renderStatusCards(statusData);
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }

        // Load and render data source status
        async function loadAndRenderDataSourceStatus(forceServer) {
            if (!forceServer && isFileProtocol) {
                const embedded = loadEmbeddedData();
                if (embedded && embedded.fetchStatus) {
                    renderDataSourceStatus(embedded.fetchStatus);
                    return;
                }
            }

            if (isFileProtocol) return;

            try {
                const response = await fetch('fetch_status.json');
                if (!response.ok) return;
                const data = await response.json();
                renderDataSourceStatus(data);
            } catch (error) {
                console.error('Error loading fetch status:', error);
            }
        }

        // Load and render all figures - always try embedded first, then server
        async function loadAndRenderFigures(forceServer) {
            if (!forceServer && isFileProtocol) {
                const embedded = loadEmbeddedData();
                if (embedded && embedded.figures) {
                    return renderFigures(embedded.figures);
                }
            }

            if (isFileProtocol) return false;

            try {
                const response = await fetch('figures.json');
                if (!response.ok) return false;
                const figuresData = await response.json();
                return renderFigures(figuresData);
            } catch (error) {
                console.error('Error loading figures:', error);
                return false;
            }
        }

        // Load metadata - always try embedded first, then server
        async function loadMetadata(forceServer) {
            if (!forceServer && isFileProtocol) {
                const embedded = loadEmbeddedData();
                if (embedded && embedded.metadata) {
                    reportMetadata = embedded.metadata;
                    predictionTime = new Date(embedded.metadata.last_update);
                    updateTimestampDisplay();
                    updateElapsed();
                    return;
                }
            }

            if (isFileProtocol) return;

            try {
                const response = await fetch('metadata.json');
                if (!response.ok) return;
                const metadata = await response.json();
                reportMetadata = metadata;
                predictionTime = new Date(metadata.last_update);
                updateTimestampDisplay();
                updateElapsed();
            } catch (error) {
                console.error('Error loading metadata:', error);
            }
        }

        // Update timestamp display
        function updateTimestampDisplay() {
            if (!predictionTime) return;

            const timestampElement = document.getElementById('timestamp');
            if (timestampElement) {
                timestampElement.textContent = predictionTime.toLocaleString('en-US', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    timeZoneName: 'short'
                });
                if (reportTimezone && reportTimezone.toLowerCase() !== 'local') {
                    timestampElement.textContent = predictionTime.toLocaleString('en-US', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        timeZone: reportTimezone,
                        timeZoneName: 'short'
                    });
                }
            }
        }

        // Update elapsed time counter
        function updateElapsed() {
            if (!predictionTime) return;

            const now = new Date();
            const diff = Math.floor((now - predictionTime) / 1000);
            const hours = Math.floor(diff / 3600).toString().padStart(2, '0');
            const minutes = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
            const seconds = (diff % 60).toString().padStart(2, '0');
            const elapsedElement = document.getElementById('elapsed');
            if (elapsedElement) {
                elapsedElement.textContent = hours + ':' + minutes + ':' + seconds;
            }
        }

        // Setup Server-Sent Events listener for real-time updates
        function setupSSE() {
            // SSE only works with HTTP, not file:// protocol
            if (isFileProtocol) {
                return false;
            }

            try {
                eventSource = new EventSource('/events');

                eventSource.addEventListener('connected', function(e) {
                    console.log('[OK] Connected to live update server');
                });

                eventSource.addEventListener('update', function(e) {
                    const data = JSON.parse(e.data);
                    console.log('[IN] Update received, refreshing data...');

                    // Reload metadata first, then figures/status to keep event mapping in sync
                    (async function() {
                        await loadMetadata(true);
                        await loadAndRenderFigures(true);
                        await loadAndRenderStatus(true);
                        await loadAndRenderDataSourceStatus(true);
                        checkStaleData();
                    })();
                });

                eventSource.onerror = function(e) {
                    console.log('[ERR] SSE connection error, falling back to polling...');
                    eventSource.close();
                    eventSource = null;
                    return false;
                };

                return true;
            } catch (e) {
                console.log('SSE not available:', e);
                return false;
            }
        }

        // Initialize
        (async function() {
            const useServer = !isFileProtocol;
            await loadMetadata(useServer);
            await loadAndRenderFigures(useServer);
            await loadAndRenderStatus(useServer);
            await loadAndRenderDataSourceStatus(useServer);
            checkStaleData();
        })();
        setInterval(updateElapsed, 1000);
        setInterval(checkStaleData, 15000);

        // Setup refresh based on protocol
        if (isFileProtocol) {
            // For file:// protocol, reload page every 30 seconds
            console.log('[AUTO] Setting up page auto-reload every 30 seconds');
            refreshInterval = setInterval(function() {
                console.log('[TIME] Auto-reloading page...');
                window.location.reload();
            }, 30000);
        } else {
            // Try to setup SSE (only works via HTTP server)
            const sseConnected = setupSSE();
            if (!sseConnected) {
                console.log('SSE failed, falling back to polling (checking every 30 seconds)');
                setInterval(function() {
                    (async function() {
                        await loadMetadata(true);
                        await loadAndRenderFigures(true);
                        await loadAndRenderStatus(true);
                        await loadAndRenderDataSourceStatus(true);
                        checkStaleData();
                    })();
                }, 30000);
            }
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/flowbite@2.4.1/dist/flowbite.min.js"></script>
</body>
</html>"""
        # Inject report timezone value into the static script block.
        html_content = html_content.replace("{report_timezone_js}", report_timezone_js)
        html_content = html_content.replace("{source_labels_js}", source_labels_js)
        # Write HTML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
