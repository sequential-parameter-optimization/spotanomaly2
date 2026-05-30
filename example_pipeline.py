# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal example: run the batch anomaly-detection pipeline from Python.

Downloads + processes data when it is missing/stale, trains a model when none
is ready, then runs detection. Each stage is an explicit, composable call on
``Pipeline``.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from spotanomaly2.application.config import load_config  # noqa: E402
from spotanomaly2.application.pipeline import Pipeline  # noqa: E402

parser = argparse.ArgumentParser(description="Run the anomaly-detection pipeline.")
parser.add_argument(
    "--config",
    type=Path,
    default=Path("config.yaml"),
    help="Path to the config YAML file (default: config.yaml)",
)
args = parser.parse_args()

config = load_config(args.config)
pipeline = Pipeline(config)

if not pipeline.data_is_ready():
    pipeline.download()
    # pipeline.download(ignore_cache=True)

pipeline.process()

if not pipeline.model_is_ready():
    pipeline.train()

results = pipeline.detect()


# pipeline.tune()
