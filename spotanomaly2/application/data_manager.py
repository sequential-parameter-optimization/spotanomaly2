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

        metadata = self._build_raw_metadata(panel_data, version_dir, row_counts, live)
        metadata_path = version_dir / "meta.json"
        storage.save_raw_metadata(metadata, metadata_path)
        self.logger.info(f"Saved {'live ' if live else ''}metadata to {metadata_path}")

    def _build_raw_metadata(
        self,
        panel_data: dict[str, pd.DataFrame],
        version_dir: Path,
        row_counts: dict[str, int],
        live: bool,
    ) -> dict[str, Any]:
        """Build the metadata dict written alongside saved raw panel parquets."""
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

        metadata: dict[str, Any] = {
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

        return metadata

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
            existing_df, bootstrap_source = self._resolve_existing_live_panel(panel_id, processed_dir, live_dir)
            self._merge_and_save_live_panel(panel_id, existing_df, new_df, bootstrap_source, live_dir)

    def _resolve_existing_live_panel(
        self,
        panel_id: str,
        processed_dir: Path,
        live_dir: Path,
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Decide which dataframe to merge new live data into for a single panel."""
        try:
            existing_df = storage.load_panel_parquet(live_dir, panel_id)
        except FileNotFoundError:
            return self._bootstrap_live_from_baseline(panel_id, processed_dir)

        self.logger.info(f"Found existing live data for panel {panel_id} with {len(existing_df)} rows")

        refreshed_df, bootstrap_source = self._maybe_refresh_from_baseline(existing_df, panel_id, processed_dir)
        if refreshed_df is not None:
            existing_df = refreshed_df

        existing_df = self._maybe_fix_gaps_with_full(existing_df, panel_id, processed_dir)
        return existing_df, bootstrap_source

    def _maybe_refresh_from_baseline(
        self,
        existing_df: pd.DataFrame,
        panel_id: str,
        processed_dir: Path,
    ) -> tuple[pd.DataFrame | None, str | None]:
        """Swap live for baseline when schema changed or live trails baseline by >1h."""
        try:
            baseline_df = storage.load_panel_parquet(processed_dir, panel_id)
        except FileNotFoundError:
            return None, None

        weight_suffix = self.config.get("process", {}).get("imputation", {}).get("weight_suffix", "__weight")
        baseline_cols = {c for c in baseline_df.columns if not c.endswith(weight_suffix)}
        existing_cols = {c for c in existing_df.columns if not c.endswith(weight_suffix)}

        if baseline_cols != existing_cols:
            self.logger.warning(
                f"Column schema mismatch for panel {panel_id}: "
                f"baseline has {baseline_cols - existing_cols} extra column(s), "
                f"live has {existing_cols - baseline_cols} extra column(s). "
                "Re-bootstrapping live data from baseline to ensure complete history."
            )
            return baseline_df, "baseline_schema_update"

        if baseline_df.index.max() > existing_df.index.max() + pd.Timedelta(hours=1):
            # User likely re-ran the full pipeline; merging stale live with the new
            # incremental window would leave the intervening period as a gap.
            self.logger.warning(
                f"Live data for panel {panel_id} is stale: "
                f"ends at {existing_df.index.max()}, baseline ends at {baseline_df.index.max()}. "
                "Re-bootstrapping from baseline to avoid creating a gap."
            )
            return baseline_df, "baseline_stale_live"

        return None, None

    def _maybe_fix_gaps_with_full(
        self,
        existing_df: pd.DataFrame,
        panel_id: str,
        processed_dir: Path,
    ) -> pd.DataFrame:
        """Replace existing data with full processed data if it has fixable gaps."""
        large_gaps = self._gaps_over_one_hour(existing_df)
        if large_gaps.empty:
            return existing_df

        self.logger.warning(f"Detected {len(large_gaps)} gap(s) > 1 hour in existing live data for panel {panel_id}")

        try:
            full_df = storage.load_panel_parquet(processed_dir, panel_id)
        except FileNotFoundError:
            error_msg = (
                f"CRITICAL: Cannot verify gaps in live data for panel {panel_id}. "
                f"Full processed data not found. Run the full pipeline first:\n"
                f"  spotanomaly2 --config <your_config.yaml> all --skip-download\n"
                f"Or from Python: pipeline.run_all(skip_download=True)"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        fixable, natural = self._categorize_gaps(existing_df, full_df, large_gaps)

        if fixable:
            self.logger.info(
                f"Loading full processed data for panel {panel_id} "
                f"({len(full_df)} rows) to fix {len(fixable)} processing artifact(s)"
            )
            return full_df
        if natural:
            self.logger.info(f"All {len(natural)} gap(s) are natural sensor outages - keeping existing data")
        return existing_df

    def _categorize_gaps(
        self,
        existing_df: pd.DataFrame,
        full_df: pd.DataFrame,
        large_gaps: pd.Series,
    ) -> tuple[list, list]:
        """Split gaps into (fixable_artifacts, natural_outages) using full data as truth."""
        full_gaps = self._gaps_over_one_hour(full_df)
        fixable: list = []
        natural: list = []

        for gap_idx, gap_duration in large_gaps.items():
            gap_start = existing_df.index[existing_df.index.get_loc(gap_idx) - 1]
            gap_end = gap_idx

            if self._gap_present_in_full(gap_start, gap_end, full_df, full_gaps):
                natural.append((gap_start, gap_end, gap_duration))
                self.logger.info(f"  Natural gap (sensor outage): {gap_start} -> {gap_end} ({gap_duration})")
            else:
                fixable.append((gap_start, gap_end, gap_duration))
                self.logger.warning(f"  Fixable gap (processing artifact): {gap_start} -> {gap_end} ({gap_duration})")

        return fixable, natural

    @staticmethod
    def _gap_present_in_full(
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
        full_df: pd.DataFrame,
        full_gaps: pd.Series,
    ) -> bool:
        """True if a gap with the same endpoints (within 1h tolerance) exists in full data."""
        for full_gap_idx in full_gaps.index:
            full_gap_start = full_df.index[full_df.index.get_loc(full_gap_idx) - 1]
            full_gap_end = full_gap_idx
            if (
                abs((gap_start - full_gap_start).total_seconds()) < 3600
                and abs((gap_end - full_gap_end).total_seconds()) < 3600
            ):
                return True
        return False

    def _bootstrap_live_from_baseline(
        self,
        panel_id: str,
        processed_dir: Path,
    ) -> tuple[pd.DataFrame, str]:
        """Initial bootstrap when no live data exists yet."""
        self.logger.warning(f"No live data found for panel {panel_id}. Attempting to bootstrap...")

        try:
            baseline_df = storage.load_panel_parquet(processed_dir, panel_id)
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

        self.logger.info(
            f"Bootstrapped panel {panel_id} from baseline "
            f"({len(baseline_df)} rows: {baseline_df.index.min()} to {baseline_df.index.max()})"
        )
        return baseline_df, "baseline"

    def _merge_and_save_live_panel(
        self,
        panel_id: str,
        existing_df: pd.DataFrame | None,
        new_df: pd.DataFrame,
        bootstrap_source: str | None,
        live_dir: Path,
    ) -> None:
        """Merge new data into existing live data and persist."""
        if existing_df is None:
            # Safeguard - prerequisite checks should prevent reaching this branch.
            self.logger.warning(
                f"No existing data available for panel {panel_id}, saving only new data. "
                f"This may result in incomplete baseline."
            )
            storage.save_panel_parquet(new_df, live_dir, panel_id)
            self.logger.info(f"Saved live processed panel {panel_id} with {len(new_df)} rows (new file)")
            return

        # combine_first preserves existing non-null values but lets new data backfill
        # previously NaN cells (important for late-arriving exogenous series).
        # concat+drop_duplicates(keep="first") would freeze first-seen NaNs forever.
        merged_df = existing_df.combine_first(new_df).sort_index()

        merge_log = f"Merged data: {len(existing_df)} existing + {len(new_df)} new = {len(merged_df)} total rows"
        if bootstrap_source == "baseline":
            merge_log += " (bootstrapped from baseline)"
        self.logger.info(merge_log)

        self._warn_on_remaining_gaps(merged_df, panel_id)

        storage.save_panel_parquet(merged_df, live_dir, panel_id)
        self.logger.info(f"Saved live processed panel {panel_id} with {len(merged_df)} rows")

    def _warn_on_remaining_gaps(self, merged_df: pd.DataFrame, panel_id: str) -> None:
        """Log a warning per remaining >1h gap in the merged data."""
        final_gaps = self._gaps_over_one_hour(merged_df)
        if final_gaps.empty:
            return
        self.logger.warning(f"Warning: {len(final_gaps)} gap(s) remain in merged data for panel {panel_id}")
        for idx, gap in final_gaps.items():
            prev = merged_df.index[merged_df.index.get_loc(idx) - 1]
            self.logger.warning(f"  Gap: {prev} -> {idx} ({gap})")

    @staticmethod
    def _gaps_over_one_hour(df: pd.DataFrame) -> pd.Series:
        """Return timestamp diffs > 1h, indexed by the gap's end timestamp."""
        time_diff = df.index.to_series().diff()
        return time_diff[time_diff > pd.Timedelta(hours=1)]

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
        results: dict[
            str,
            tuple[
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame | None,
                dict[str, pd.DataFrame] | None,
            ],
        ],
        timestamp: str | None = None,
        live_mode: bool = False,
    ) -> Path:
        """Save detection results to a timestamped or live directory."""
        base_results_dir = Path(self.config["paths"]["results_dir"])

        if live_mode:
            results_dir = base_results_dir / "live"
        else:
            if timestamp is None:
                timestamp = generate_timestamp()
            results_dir = base_results_dir / timestamp

        storage.ensure_dir(results_dir)

        for panel_id, (scores_df, flags_df, forecast_df, contributions_df, per_channel) in results.items():
            scores_filename = f"panel_{panel_id}_scores.csv"
            flags_filename = f"panel_{panel_id}_flags.csv"
            forecast_filename = f"panel_{panel_id}_forecast.csv"
            scores_df.to_csv(results_dir / scores_filename)
            flags_df.to_csv(results_dir / flags_filename)
            forecast_df.to_csv(results_dir / forecast_filename)
            if contributions_df is not None:
                contrib_filename = f"panel_{panel_id}_contributions.parquet"
                contributions_df.to_parquet(results_dir / contrib_filename)
            # Per-channel detection results
            if per_channel is not None:
                for key in ("scores", "flags", "flags_combined", "thresholds"):
                    if key in per_channel:
                        fname = f"panel_{panel_id}_per_channel_{key}.csv"
                        per_channel[key].to_csv(results_dir / fname)
            self.logger.info(f"Saved results for panel {panel_id} to {results_dir}")

        return results_dir
