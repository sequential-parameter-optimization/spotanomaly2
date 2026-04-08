#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from spotanomaly2.domain import imputation_methods


def load_all_processed_data() -> Dict[str, pd.DataFrame]:
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    panel_data = {}

    for file in sorted(list(processed_dir.glob('*_processed.csv')) +
                       list(processed_dir.glob('panel_*.parquet'))):
        if '_processed' in file.stem:
            panel_id = file.stem.replace('_processed', '')
        else:
            panel_id = file.stem

        if panel_id in panel_data and file.suffix == '.csv':
            continue

        try:
            if file.suffix == '.parquet':
                df = pd.read_parquet(file)
                if 'time' in df.columns:
                    df = df.set_index('time')
            else:
                df = pd.read_csv(file, index_col=0, parse_dates=[0])

            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

            panel_data[panel_id] = df
            print(f"✓ Loaded {panel_id}: {len(df)} samples, {len(df.columns)} columns")
        except Exception as e:
            print(f"✗ Failed to load {file.name}: {e}")

    if not panel_data:
        raise ValueError("No processed data found")

    return panel_data


def inject_gaps_in_series(
    series: pd.Series,
    gap_size: int = 5,
    num_gaps: int = 10,
    random_seed: int = 42
) -> Tuple[pd.Series, List[Tuple[int, int]], np.ndarray]:
    np.random.seed(random_seed)
    series_with_gaps = series.copy()

    if series.dtype not in [np.float64, np.float32, np.int64, np.int32]:
        return series, [], np.array([])

    valid_indices = np.where(~series_with_gaps.isna())[0]
    if len(valid_indices) < gap_size * num_gaps:
        return series, [], np.array([])

    gap_positions = []
    original_values = []

    for _ in range(num_gaps * 5):
        if len(gap_positions) >= num_gaps:
            break

        start_idx = np.random.randint(gap_size, len(series_with_gaps) - gap_size)
        overlaps = any(
            (start_idx >= pos[0] - gap_size) and (start_idx <= pos[1] + gap_size)
            for pos in gap_positions
        )

        if not overlaps:
            end_idx = start_idx + gap_size
            original_vals = series_with_gaps.iloc[start_idx:end_idx].values.copy()

            if not np.isnan(original_vals).any():
                series_with_gaps.iloc[start_idx:end_idx] = np.nan
                gap_positions.append((start_idx, end_idx))
                original_values.append(original_vals)

    if original_values:
        original_values_array = np.concatenate(original_values)
    else:
        original_values_array = np.array([])

    return series_with_gaps, gap_positions, original_values_array


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    mask = np.abs(y_true) > 1e-10
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def test_method_on_column(
    series: pd.Series,
    method: str,
    gap_positions: List[Tuple[int, int]],
    original_values: np.ndarray,
    **params
) -> Dict[str, object]:
    try:
        start_time = time.time()
        imputed = imputation_methods.impute_series(series, method=method, **params)
        elapsed = time.time() - start_time

        imputed_values = []
        for start_idx, end_idx in gap_positions:
            imputed_values.extend(imputed.iloc[start_idx:end_idx].values)

        if imputed_values:
            imputed_values = np.array(imputed_values)
            metrics = calculate_metrics(original_values, imputed_values)
        else:
            metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        return {
            'success': True,
            'time_ms': elapsed * 1000,
            'metrics': metrics,
            'error': None,
        }
    except Exception as e:
        return {
            'success': False,
            'time_ms': np.nan,
            'metrics': {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan},
            'error': str(e)[:80],
        }


def benchmark_panel(
    panel_id: str,
    df: pd.DataFrame,
    gap_size: int = 5,
    num_gaps: int = 10,
) -> pd.DataFrame:
    print(f"\n{'=' * 80}")
    print(f"Benchmarking {panel_id}")
    print(f"{'=' * 80}")

    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    methods_to_test = {
        'mean': {},
        'forward_fill': {},
        'backward_fill': {},
        'linear_interpolation': {},
        'spline_interpolation': {},
        'knn_temporal': {'n_neighbors': 5},
        'seasonal': {'period': 288},
        'rolling_mean': {'window': 10},
    }

    try:
        from sklearn.impute import IterativeImputer  # noqa: F401
        methods_to_test['iterative'] = {'max_iter': 10}
    except Exception:
        pass

    try:
        from sklearn.impute import KNNImputer  # noqa: F401
        methods_to_test['knn_sklearn'] = {'n_neighbors': 5}
    except Exception:
        pass

    for col_idx, col in enumerate(numeric_cols, 1):
        series = df[col].copy()

        if series.isna().sum() > len(series) * 0.3:
            print(f"  Skipping {col} (too much missing data)")
            continue

        series_with_gaps, gap_positions, original_values = inject_gaps_in_series(
            series, gap_size=gap_size, num_gaps=num_gaps
        )

        if len(gap_positions) == 0:
            print(f"  Skipping {col} (couldn't inject gaps)")
            continue

        print(f"  [{col_idx}/{len(numeric_cols)}] {col:30} ", end='', flush=True)

        for method_name, params in methods_to_test.items():
            result = test_method_on_column(
                series_with_gaps, method_name, gap_positions, original_values, **params
            )

            results.append({
                'panel': panel_id,
                'column': col,
                'method': method_name,
                'success': result['success'],
                'MAE': result['metrics']['MAE'],
                'RMSE': result['metrics']['RMSE'],
                'MAPE': result['metrics']['MAPE'],
                'time_ms': result['time_ms'],
                'error': result['error'],
            })

            print('.' if result['success'] else 'x', end='', flush=True)

        print()

    return pd.DataFrame(results)


def summarize_results(all_results: pd.DataFrame) -> pd.DataFrame:
    summary = all_results[all_results['success']].groupby('method').agg({
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std'],
        'time_ms': ['mean'],
        'column': 'count'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'column_count': 'num_tests'})

    return summary.sort_values('MAE_mean')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare imputation methods on real sensor data (no plots, no files)'
    )
    parser.add_argument('--gap-size', type=int, default=5)
    parser.add_argument('--num-gaps', type=int, default=10)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("IMPUTATION METHODS COMPARISON")
    print("=" * 80)
    print(f"Gap size: {args.gap_size} samples")
    print(f"Gaps per column: {args.num_gaps}\n")

    print("Loading processed data...")
    panel_data = load_all_processed_data()
    print(f"✓ Loaded {len(panel_data)} panels\n")

    all_results = []
    for panel_id, df in panel_data.items():
        panel_results = benchmark_panel(
            panel_id, df, gap_size=args.gap_size, num_gaps=args.num_gaps
        )
        all_results.append(panel_results)

    results_df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary = summarize_results(results_df)
    print(summary)

    if len(results_df) > 0:
        best_by_mae = results_df[results_df['success']].groupby('method')['MAE'].mean().idxmin()
        print("\nBest by MAE:", best_by_mae)

    print("\n" + "=" * 80)
    print("✓ Comparison complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
