# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal example: run the live anomaly-detection pipeline from Python.

On first run (or when data/models are older than 7 days), this will
automatically download, process, and train before switching to live mode.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from spotanomaly2.application.config import load_default_config  # noqa: E402
from spotanomaly2.application.pipeline import Pipeline  # noqa: E402
from spotanomaly2.dashboard import LiveMonitor  # noqa: E402

config = load_default_config()
pipeline = Pipeline(config)

if not (pipeline.data_is_ready() and pipeline.model_is_ready()):
    print("Baseline data/models missing or older than 7 days — running full pipeline...")
    print("This may take a few minutes.\n")
    pipeline.download()
    pipeline.process()
    pipeline.train()
    pipeline.detect()
    print("\nBaseline created. Switching to live mode...\n")

results = LiveMonitor(config).run_once()

for panel_id, (scores_df, flags_df, forecast_df, _contrib) in results.items():
    status = "anomaly detected" if flags_df["anomaly_flag"].iloc[-1] else "all clear"
    print(f"\n── Panel {panel_id} ({status}) ──")
    print(f"  scores:   {scores_df.shape}")
    print(f"  flags:    {flags_df.shape}")
    print(f"  forecast: {forecast_df.shape}")
    print(flags_df.tail(3))
