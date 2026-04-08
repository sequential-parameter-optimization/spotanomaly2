# SpotAnomaly

Streamlined anomaly detection system for time series data with dual training paradigms (batch and online learning).

## Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

## Installation

### Using uv (Recommended)

Install uv:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Set up the project:

```bash
uv sync
```

This command automatically installs Python 3.11, creates a virtual environment, and installs all dependencies.

### Using pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Usage

### Configuration

Choose your training paradigm in `config/default.yaml`:

```yaml
model:
  paradigm: "batch"  # or "online"
```

- **batch**: Traditional batch training with sklearn/LightGBM (faster, trains once)
- **online**: Streaming online learning with river (incremental updates)

### Running the Pipeline

Run all steps a first time:

```bash
uv run python -m src all
```

Run all steps but skip the download:

```bash
uv run python -m src all --skip-download
```

Or run individual steps:

```bash
uv run python -m src process  # Process raw data
uv run python -m src train    # Train forecasting models
uv run python -m src detect   # Detect anomalies
```

Run predictions with a pre-trained model:

```bash
uv run python -m src all --predict-only --model 20251229_172126
```

### Viewing the result
You can explore, chart, and analyze the processed and modeled data interactively in [notebooks/analyze_combined.ipynb](notebooks/analyze_combined.ipynb).


### Running live mode

```bash
uv run python -m src live --interval 5
```

Then open [the live report](data/results/live/report.html)

## Development

Install with development dependencies:

```bash
uv sync --all-extras
```