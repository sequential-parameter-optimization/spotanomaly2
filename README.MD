# SpotAnomaly2

Forecast-based anomaly detection for multi-channel time-series data. Each channel is
forecast by a per-channel model; residuals between forecast and actual are scored, and
points whose residuals exceed a calibrated threshold are flagged as anomalies. The same
pipeline runs as a one-shot batch job or as a long-running live monitor.

## Requirements

- Python **3.13+** (`uv` installs a matching interpreter for you; `.python-version` pins it)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

Install uv:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Homebrew
brew install uv
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Set up the project (creates the virtualenv and installs everything):

```bash
uv sync                 # runtime dependencies
uv sync --all-groups    # + dev tools (ruff, ty, reuse, pytest, jupyter)
```

## Repository layout

```
spotanomaly2/           # package
  application/          # config loading + Pipeline orchestration
  domain/               # forecasting, anomaly detection, model training
  infrastructure/       # logging, IO, adapters
  dashboard/            # live monitor + HTML report server
  examples/
  __main__.py → main.py # CLI entry point
config/                 # default.yaml (bundled default) + config.yaml
tests/                  # pytest suite
notebooks/              # exploratory analysis notebooks
data/                   # raw / processed / models / results (generated)
```

## Configuration

The CLI loads [config/default.yaml](config/default.yaml) unless you pass `--config`:

```bash
uv run python -m spotanomaly2 detect --config ./config/config.yaml
```

Key sections in the config: `panels` (channels per panel), `paths` (data/model/result
dirs), `fetch` / `process` (download & preprocessing), `train` (split + per-channel
models), `detect` (`hist_window`, threshold, `scorer_fit_scope`), `tune` (SpotOptim
hyperparameter search) and `report` (live HTML output). See `default.yaml` for the full,
commented schema.

## Usage

Run via the module (`python -m spotanomaly2 <command>`); an installed console script
named `spotanomaly2` is also available (e.g. `uv run spotanomaly2 detect`).

| Command | What it does |
|---|---|
| `download` | Fetch raw data from the configured API and save as Parquet |
| `process`  | Convert, resample and preprocess raw data |
| `tune`     | Search per-channel forecaster hyperparameters with SpotOptim (optional) |
| `train`    | Train per-channel forecasting models |
| `detect`   | Score residuals with a trained model and flag anomalies |
| `live`     | Download → process → predict with an existing model (no training) |

Typical end-to-end run:

```bash
uv run python -m spotanomaly2 download
uv run python -m spotanomaly2 process
uv run python -m spotanomaly2 tune      # optional
uv run python -m spotanomaly2 train
uv run python -m spotanomaly2 detect
```

Useful options:

```bash
# Run any step against a specific config
uv run python -m spotanomaly2 process --config ./config/config.yaml

# Detect with a specific trained model (default: most recent)
uv run python -m spotanomaly2 detect --model 20250115_143022

# Pin the raw-data version used by process / train / detect (default: most recent)
uv run python -m spotanomaly2 train --raw-data-version 20260105_174531

# Tune a single panel/channel, overriding trial counts
uv run python -m spotanomaly2 tune --panel 1 --channel channel_1_ph --n-trials 50 --n-initial 10
```

Run `uv run python -m spotanomaly2 <command> --help` for the full option list of any command.

### Live monitoring

Run once, or continuously every N minutes:

```bash
uv run python -m spotanomaly2 live                 # single pass, then exit
uv run python -m spotanomaly2 live --interval 5    # re-run every 5 minutes
```

Live mode serves an auto-updating report at [data/results/live/report.html](data/results/live/report.html).

### Analyzing results

Explore processed data, forecasts and detected anomalies interactively in
[notebooks/analyze_combined.ipynb](notebooks/analyze_combined.ipynb) and the other
notebooks under [notebooks/](notebooks/).

## Development

```bash
uv sync --all-groups        # install dev tools

uv run ruff check .         # lint
uv run ruff format .        # format
uv run ty check .           # type-check
uv run reuse lint           # SPDX/license headers
uv run pytest               # tests
```

## License

AGPL-3.0-or-later. This project is [REUSE](https://reuse.software/)-compliant — every
source file carries an SPDX header (see [REUSE.toml](REUSE.toml) and [LICENSES/](LICENSES/)).
