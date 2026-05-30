# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Joiner-side tests for exogenous source A.

The fetcher's HTTP/cache machinery requires real upstream credentials and lives
under integration tests. These tests cover ``ExogenousAJoiner`` in isolation:
write a fixture parquet to a ``tmp_path`` cache_dir, instantiate the joiner,
and assert the panel DataFrame gets the expected
``exogenous_<yaml_name>_<panel_source>`` columns appended. Direct construction
here doesn't pass ``source_name``, so the joiner falls back to its default
identity ``"a"`` — matching what the registry injects for the bundled source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spotanomaly2.domain.exogenous.source_a import ExogenousAJoiner


def _write_source_parquet(cache_dir: Path, source_name: str, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{source_name}.parquet")


def _panel_df(start: str = "2025-01-01", periods: int = 12) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    return pd.DataFrame({"sensor": [1.0] * periods}, index=idx)


def _source_df(start: str = "2025-01-01", periods: int = 12, value: float = 7.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    return pd.DataFrame({"flow": [value] * periods}, index=idx)


@pytest.fixture
def parent_cfg():
    return {"process": {"resample": {"freq": "5min"}}}


class TestEmptyInputs:
    def test_empty_panel_dict_returns_empty(self, parent_cfg, tmp_path):
        source_cfg = {"cache_dir": str(tmp_path), "panel_sources": {"1": ["flow_a"]}}
        joiner = ExogenousAJoiner(source_cfg, parent_cfg)
        assert joiner.join_into_panels({}) == {}

    def test_missing_cache_for_panel_skips_with_warning(self, parent_cfg, tmp_path):
        source_cfg = {"cache_dir": str(tmp_path), "panel_sources": {"1": ["flow_a"]}}
        joiner = ExogenousAJoiner(source_cfg, parent_cfg)
        panel_data = {"1": _panel_df()}
        # No parquet on disk → joiner returns input unchanged (warning logged).
        result = joiner.join_into_panels(panel_data)
        # Panel kept; no exogenous columns added (cache missing).
        assert "exogenous_a_flow_a" not in result["1"].columns


class TestSingleSourceJoin:
    def test_single_column_source_gets_renamed_without_suffix(self, parent_cfg, tmp_path):
        # Single measurement column → column renamed to exogenous_<source>.
        _write_source_parquet(tmp_path, "flow_a", _source_df(value=7.0))
        source_cfg = {"cache_dir": str(tmp_path), "panel_sources": {"1": ["flow_a"]}}
        joiner = ExogenousAJoiner(source_cfg, parent_cfg)

        result = joiner.join_into_panels({"1": _panel_df()})
        assert "exogenous_a_flow_a" in result["1"].columns
        # Single col → no "_<colname>" suffix.
        assert all(c.startswith("exogenous_a_flow_a") for c in result["1"].columns if c.startswith("exogenous_"))
        # Original sensor column preserved.
        assert "sensor" in result["1"].columns

    def test_metric_column_excluded_from_join(self, parent_cfg, tmp_path):
        # Source has both a measurement and a 'metric' column; only the measurement should land.
        idx = pd.date_range("2025-01-01", periods=12, freq="5min", tz="UTC")
        df = pd.DataFrame({"flow": [7.0] * 12, "metric": ["something"] * 12}, index=idx)
        _write_source_parquet(tmp_path, "flow_a", df)
        source_cfg = {"cache_dir": str(tmp_path), "panel_sources": {"1": ["flow_a"]}}
        joiner = ExogenousAJoiner(source_cfg, parent_cfg)

        result = joiner.join_into_panels({"1": _panel_df()})
        # 'metric' must not propagate as an exogenous column.
        assert not any("metric" in c for c in result["1"].columns)
        assert "exogenous_a_flow_a" in result["1"].columns


class TestPerPanelRouting:
    def test_different_panels_get_different_sources(self, parent_cfg, tmp_path):
        _write_source_parquet(tmp_path, "src_x", _source_df(value=10.0))
        _write_source_parquet(tmp_path, "src_y", _source_df(value=20.0))
        source_cfg = {
            "cache_dir": str(tmp_path),
            "panel_sources": {"1": ["src_x"], "2": ["src_y"]},
        }
        joiner = ExogenousAJoiner(source_cfg, parent_cfg)

        panel_data = {"1": _panel_df(), "2": _panel_df()}
        result = joiner.join_into_panels(panel_data)

        assert "exogenous_a_src_x" in result["1"].columns
        assert "exogenous_a_src_x" not in result["2"].columns
        assert "exogenous_a_src_y" in result["2"].columns
        assert "exogenous_a_src_y" not in result["1"].columns

    def test_panel_without_mapping_gets_passthrough(self, parent_cfg, tmp_path):
        _write_source_parquet(tmp_path, "src_x", _source_df(value=10.0))
        source_cfg = {"cache_dir": str(tmp_path), "panel_sources": {"1": ["src_x"]}}
        joiner = ExogenousAJoiner(source_cfg, parent_cfg)

        panel_data = {"1": _panel_df(), "2": _panel_df()}
        original_panel_2 = panel_data["2"].copy()
        result = joiner.join_into_panels(panel_data)

        assert "exogenous_a_src_x" in result["1"].columns
        # Panel 2 has no mapping → passthrough.
        pd.testing.assert_frame_equal(result["2"], original_panel_2)
