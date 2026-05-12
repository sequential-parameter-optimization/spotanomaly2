"""Data load/save operations for raw, processed, and detection results."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spotanomaly2.infrastructure import logging, storage
from spotanomaly2.infrastructure.storage import generate_timestamp


class DataManager:
    """Handles all disk I/O for raw data, processed data, and detection results."""

    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.get_logger("DataManager")

    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load raw data from disk."""
        raw_dir = Path(self.config["paths"]["raw_dir"])
        panels = self.config["panels"]["panel_ids"]
        version = self.config["paths"].get("raw_data_version")

        if version:
            self.logger.info(f"Loading raw data from version: {version}")
        else:
            self.logger.info("Loading raw data from latest version")

        panel_data = {}
        for panel_id in panels:
            self.logger.info(f"Loading raw data for panel {panel_id}...")
            df = storage.load_panel_parquet_versioned(raw_dir, panel_id, version)
            panel_data[panel_id] = df
            self.logger.info(f"Loaded {len(df)} rows for panel {panel_id}")
        return panel_data

    def save_raw_data(self, panel_data: dict[str, pd.DataFrame], live: bool = False) -> None:
        """Save raw data to disk in a versioned or live directory with metadata."""
        raw_dir = Path(self.config["paths"]["raw_dir"])
        storage.ensure_dir(raw_dir)

        if live:
            version_dir = raw_dir / "live"
        else:
            version_dir = raw_dir / generate_timestamp()
        storage.ensure_dir(version_dir)

        self.logger.info(f"Saving {'live ' if live else ''}raw data to: {version_dir}")

        row_counts = {}
        for panel_id, df in panel_data.items():
            storage.save_panel_parquet(df, version_dir, panel_id)
            row_counts[panel_id] = len(df)
            self.logger.info(
                f"Saved {'live ' if live else ''}panel {panel_id} to {version_dir / f'panel_{panel_id}.parquet'}"
            )

        start_date = self.config["fetch"]["start_date"]
        end_date = self.config["fetch"].get("end_date")
        if end_date is None:
            end_date = datetime.now().isoformat() + "+00:00"

        actual_start = None
        actual_end = None
        for df in panel_data.values():
            if len(df) > 0:
                df_start = df.index.min()
                df_end = df.index.max()
                if actual_start is None or df_start < actual_start:
                    actual_start = df_start
                if actual_end is None or df_end > actual_end:
                    actual_end = df_end

        if actual_start is not None and hasattr(actual_start, "isoformat"):
            actual_start = actual_start.isoformat()
        if actual_end is not None and hasattr(actual_end, "isoformat"):
            actual_end = actual_end.isoformat()

        metadata = {
            "start_date": start_date,
            "end_date": end_date,
            "panel_ids": list(panel_data.keys()),
            "row_counts": row_counts,
        }
        if live:
            metadata["last_update"] = datetime.now().isoformat()
        else:
            metadata["download_timestamp"] = version_dir.name
        if actual_start is not None:
            metadata["actual_start"] = actual_start
        if actual_end is not None:
            metadata["actual_end"] = actual_end

        metadata_path = version_dir / "meta.json"
        storage.save_raw_metadata(metadata, metadata_path)
        self.logger.info(f"Saved {'live ' if live else ''}metadata to {metadata_path}")

    def load_processed_data(self) -> dict[str, pd.DataFrame]:
        """Load processed data from disk."""
        processed_dir = Path(self.config["paths"]["processed_dir"])
        panels = self.config["panels"]["panel_ids"]
        panel_data = {}
        for panel_id in panels:
            self.logger.info(f"Loading processed data for panel {panel_id}...")
            df = storage.load_panel_parquet(processed_dir, panel_id)
            panel_data[panel_id] = df
            self.logger.info(f"Loaded {len(df)} rows for panel {panel_id}")
        return panel_data

    def save_processed_data(self, processed_data: dict[str, pd.DataFrame]) -> None:
        """Save processed data to disk."""
        processed_dir = Path(self.config["paths"]["processed_dir"])
        storage.ensure_dir(processed_dir)
        for panel_id, df in processed_data.items():
            storage.save_panel_parquet(df, processed_dir, panel_id)
            self.logger.info(f"Saved processed panel {panel_id} with {len(df)} rows")

    def save_processed_data_live(self, processed_data: dict[str, pd.DataFrame]) -> None:
        """Save processed data to live directory, merging with existing data.

        Live directory is the canonical source for continuous data.
        The baseline processed/ directory is only used for initial bootstrap.

        Includes automatic gap detection and fixing to ensure data continuity.
        """
        processed_dir = Path(self.config["paths"]["processed_dir"])
        live_dir = processed_dir / "live"
        storage.ensure_dir(live_dir)

        for panel_id, new_df in processed_data.items():
            existing_df = None
            bootstrap_source = None

            try:
                existing_df = storage.load_panel_parquet(live_dir, panel_id)
                self.logger.info(f"Found existing live data for panel {panel_id} with {len(existing_df)} rows")

                # If schema changed (e.g. new EXOGENOUS columns), refresh live history
                # from baseline to avoid long NaN spans for newly introduced targets.
                try:
                    baseline_df_check = storage.load_panel_parquet(processed_dir, panel_id)
                    weight_suffix = (
                        self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")
                    )
                    baseline_data_cols = {c for c in baseline_df_check.columns if not c.endswith(weight_suffix)}
                    existing_data_cols = {c for c in existing_df.columns if not c.endswith(weight_suffix)}
                    if baseline_data_cols != existing_data_cols:
                        self.logger.warning(
                            f"Column schema mismatch for panel {panel_id}: "
                            f"baseline has {baseline_data_cols - existing_data_cols} extra column(s), "
                            f"live has {existing_data_cols - baseline_data_cols} extra column(s). "
                            "Re-bootstrapping live data from baseline to ensure complete history."
                        )
                        existing_df = baseline_df_check
                        bootstrap_source = "baseline_schema_update"
                    elif baseline_df_check.index.max() > existing_df.index.max() + pd.Timedelta(hours=1):
                        # Baseline is fresher than live (e.g. user just re-ran the full
                        # pipeline). Without this swap, merging stale live with the new
                        # incremental window leaves the intervening period as a gap.
                        self.logger.warning(
                            f"Live data for panel {panel_id} is stale: "
                            f"ends at {existing_df.index.max()}, baseline ends at {baseline_df_check.index.max()}. "
                            "Re-bootstrapping from baseline to avoid creating a gap."
                        )
                        existing_df = baseline_df_check
                        bootstrap_source = "baseline_stale_live"
                except FileNotFoundError:
                    pass

                time_diff = existing_df.index.to_series().diff()
                large_gaps = time_diff[time_diff > pd.Timedelta(hours=1)]

                if len(large_gaps) > 0:
                    self.logger.warning(
                        f"Detected {len(large_gaps)} gap(s) > 1 hour in existing live data for panel {panel_id}"
                    )

                    # Check if these gaps are also in full processed data (natural sensor outages)
                    # or if they're processing artifacts that can be fixed
                    try:
                        full_df = storage.load_panel_parquet(processed_dir, panel_id)
                        full_time_diff = full_df.index.to_series().diff()
                        full_gaps = full_time_diff[full_time_diff > pd.Timedelta(hours=1)]

                        # Compare gaps: which ones are fixable?
                        fixable_gaps = []
                        natural_gaps = []

                        for gap_idx, gap_duration in large_gaps.items():
                            gap_start = existing_df.index[existing_df.index.get_loc(gap_idx) - 1]
                            gap_end = gap_idx

                            # Check if this gap also exists in full data
                            # (allowing for small timestamp differences due to resampling)
                            is_natural_gap = False
                            for full_gap_idx in full_gaps.index:
                                full_gap_start = full_df.index[full_df.index.get_loc(full_gap_idx) - 1]
                                full_gap_end = full_gap_idx

                                # If gaps overlap significantly, consider it a natural gap
                                if (
                                    abs((gap_start - full_gap_start).total_seconds()) < 3600
                                    and abs((gap_end - full_gap_end).total_seconds()) < 3600
                                ):
                                    is_natural_gap = True
                                    break

                            if is_natural_gap:
                                natural_gaps.append((gap_start, gap_end, gap_duration))
                                self.logger.info(
                                    f"  Natural gap (sensor outage): {gap_start} -> {gap_end} ({gap_duration})"
                                )
                            else:
                                fixable_gaps.append((gap_start, gap_end, gap_duration))
                                self.logger.warning(
                                    f"  Fixable gap (processing artifact): {gap_start} -> {gap_end} ({gap_duration})"
                                )

                        # Only fix if there are fixable gaps
                        if len(fixable_gaps) > 0:
                            self.logger.info(
                                f"Loading full processed data for panel {panel_id} "
                                f"({len(full_df)} rows) to fix {len(fixable_gaps)} processing artifact(s)"
                            )
                            existing_df = full_df
                        elif len(natural_gaps) > 0:
                            self.logger.info(
                                f"All {len(natural_gaps)} gap(s) are natural sensor outages - keeping existing data"
                            )

                    except FileNotFoundError:
                        error_msg = (
                            f"CRITICAL: Cannot verify gaps in live data for panel {panel_id}. "
                            f"Full processed data not found. Run the full pipeline first:\n"
                            f"  spotanomaly2 --config <your_config.yaml> all --skip-download\n"
                            f"Or from Python: pipeline.run_all(skip_download=True)"
                        )
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)

            except FileNotFoundError:
                # Live data doesn't exist - try to bootstrap from processed baseline
                self.logger.warning(f"No live data found for panel {panel_id}. Attempting to bootstrap...")

                try:
                    baseline_df = storage.load_panel_parquet(processed_dir, panel_id)
                    baseline_end = baseline_df.index.max()
                    baseline_start = baseline_df.index.min()

                    # Check if we have newer raw data available that wasn't processed yet
                    raw_dir = Path(self.config["paths"]["raw_dir"])
                    try:
                        # Find newest raw data version
                        version_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name != "live"]
                        if version_dirs:
                            newest_version = max(version_dirs, key=lambda d: d.name)
                            newest_raw_df = storage.load_panel_parquet(newest_version, panel_id)
                            newest_raw_end = newest_raw_df.index.max()

                            # If raw data is significantly newer than baseline, we need to process it
                            time_gap = newest_raw_end - baseline_end
                            if time_gap > pd.Timedelta(days=1):
                                error_msg = (
                                    f"CRITICAL: Live data missing for panel {panel_id}, and baseline is outdated!\n"
                                    f"  Baseline ends at: {baseline_end}\n"
                                    f"  Raw data ends at: {newest_raw_end}\n"
                                    f"  Gap: {time_gap.days} days\n\n"
                                    f"This would cause data loss. Run the full pipeline first:\n"
                                    f"  spotanomaly2 --config <your_config.yaml> all --skip-download\n"
                                    f"Or from Python: pipeline.run_all(skip_download=True)"
                                )
                                self.logger.error(error_msg)
                                raise RuntimeError(error_msg)
                    except Exception as e:
                        self.logger.warning(f"Could not check raw data currency: {e}")

                    # Bootstrap from baseline (only if it's recent enough)
                    existing_df = baseline_df
                    bootstrap_source = "baseline"
                    self.logger.info(
                        f"Bootstrapped panel {panel_id} from baseline "
                        f"({len(existing_df)} rows: {baseline_start} to {baseline_end})"
                    )

                except FileNotFoundError:
                    error_msg = (
                        f"CRITICAL: Cannot bootstrap live data for panel {panel_id}. "
                        f"No baseline processed data found. Run the full pipeline first:\n"
                        f"  spotanomaly2 --config <your_config.yaml> all\n"
                        f"Or from Python: pipeline.run_all()\n"
                        f"This will fetch, process, train models, and create the initial baseline."
                    )
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

            if existing_df is not None:
                # Merge incrementally while preserving existing non-null values,
                # but allowing new data to backfill previously missing cells
                # (important for late-arriving external series like exogenous sources).
                #
                # Why not concat+drop_duplicates(keep="first")?
                # That strategy freezes first-seen NaNs forever on overlapping
                # timestamps. If a later incremental run brings valid values for
                # those same timestamps, they are silently discarded.
                merged_df = existing_df.combine_first(new_df).sort_index()

                merge_log = (
                    f"Merged data: {len(existing_df)} existing + {len(new_df)} new = {len(merged_df)} total rows"
                )
                if bootstrap_source == "baseline":
                    merge_log += " (bootstrapped from baseline)"
                self.logger.info(merge_log)

                # Final gap check on merged data
                time_diff = merged_df.index.to_series().diff()
                final_gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
                if len(final_gaps) > 0:
                    self.logger.warning(f"Warning: {len(final_gaps)} gap(s) remain in merged data for panel {panel_id}")
                    for idx, gap in final_gaps.items():
                        prev = merged_df.index[merged_df.index.get_loc(idx) - 1]
                        self.logger.warning(f"  Gap: {prev} -> {idx} ({gap})")

                storage.save_panel_parquet(merged_df, live_dir, panel_id)
                self.logger.info(f"Saved live processed panel {panel_id} with {len(merged_df)} rows")
            else:
                # This should not happen anymore due to prerequisite checks, but keep as safeguard
                self.logger.warning(
                    f"No existing data available for panel {panel_id}, saving only new data. "
                    f"This may result in incomplete baseline."
                )
                storage.save_panel_parquet(new_df, live_dir, panel_id)
                self.logger.info(f"Saved live processed panel {panel_id} with {len(new_df)} rows (new file)")

    def load_processed_data_live(self) -> dict[str, pd.DataFrame]:
        """Load accumulated processed data from live directory."""
        processed_dir = Path(self.config["paths"]["processed_dir"])
        live_dir = processed_dir / "live"
        panels = self.config["panels"]["panel_ids"]
        panel_data = {}
        for panel_id in panels:
            try:
                df = storage.load_panel_parquet(live_dir, panel_id)
                panel_data[panel_id] = df
                self.logger.debug(f"Loaded live processed data for panel {panel_id}: {len(df)} rows")
            except FileNotFoundError:
                self.logger.warning(f"No live processed data found for panel {panel_id}")
                panel_data[panel_id] = pd.DataFrame()
        return panel_data

    def save_detection_results(
        self,
        results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]],
        timestamp: str | None = None,
        live_mode: bool = False,
    ) -> Path:
        """Save detection results to a timestamped or live directory."""
        base_results_dir = Path(self.config["paths"]["results_dir"])
        fc_model_name = self.config["detect"]["fc_model_name"]

        if live_mode:
            results_dir = base_results_dir / "live"
        else:
            if timestamp is None:
                timestamp = generate_timestamp()
            results_dir = base_results_dir / timestamp

        storage.ensure_dir(results_dir)

        for panel_id, (scores_df, flags_df, forecast_df, contributions_df) in results.items():
            scores_filename = f"{fc_model_name}_panel_{panel_id}_scores.csv"
            flags_filename = f"{fc_model_name}_panel_{panel_id}_flags.csv"
            forecast_filename = f"{fc_model_name}_panel_{panel_id}_forecast.csv"
            scores_df.to_csv(results_dir / scores_filename)
            flags_df.to_csv(results_dir / flags_filename)
            forecast_df.to_csv(results_dir / forecast_filename)
            if contributions_df is not None:
                contrib_filename = f"{fc_model_name}_panel_{panel_id}_contributions.parquet"
                contributions_df.to_parquet(results_dir / contrib_filename)
            self.logger.info(f"Saved results for panel {panel_id} to {results_dir}")

        return results_dir
