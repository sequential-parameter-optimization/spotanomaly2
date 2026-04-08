#!/usr/bin/env python3
"""
Quick test script for imputation methods.

Usage:
    python test_imputation_methods.py --method linear_interpolation --data-path path/to/data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spotanomaly2.domain import imputation_methods


def create_synthetic_test_data(size: int = 1000, seed: int = 42) -> pd.Series:
    """Create synthetic water quality data with missing values."""
    np.random.seed(seed)

    # Create synthetic daily pattern
    t = np.arange(size)
    daily_pattern = 50 + 10 * np.sin(2 * np.pi * t / 288)  # 24h cycle at 5min intervals
    noise = np.random.normal(0, 2, size)
    trend = np.linspace(0, 20, size)

    series = daily_pattern + noise + trend

    # Add some missing values
    series = pd.Series(series, index=pd.date_range("2025-01-01", periods=size, freq="5min"))
    missing_indices = np.random.choice(len(series), size=int(0.05 * len(series)), replace=False)
    series.iloc[missing_indices] = np.nan

    return series


def load_real_data(data_path: str) -> pd.Series:
    """Load real data from CSV."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=[0])
    # Get first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in data")
    return df[numeric_cols[0]].iloc[:1000]  # Use first 1000 samples


def evaluate_imputation_method(series: pd.Series, method: str, **params) -> dict:
    """Run a single imputation method and return metrics.

    Returns:
        Dict with imputation result and performance metrics
    """
    import time

    # Skip if already imputed (for baseline)
    has_missing = series.isna().sum()
    if has_missing == 0:
        return {"method": method, "success": False, "error": "No missing values in series", "time_ms": 0}

    try:
        # Time the imputation
        start = time.time()
        imputed = imputation_methods.impute_series(series, method=method, **params)
        elapsed = time.time() - start

        # Check result
        remaining_nan = imputed.isna().sum()

        return {
            "method": method,
            "success": True,
            "missing_before": has_missing,
            "missing_after": remaining_nan,
            "time_ms": elapsed * 1000,
            "imputed_series": imputed,
        }

    except Exception as e:
        return {"method": method, "success": False, "error": str(e), "time_ms": 0}


def compare_all_methods(series: pd.Series) -> pd.DataFrame:
    """Compare all available imputation methods."""
    print("Testing all imputation methods...\n")

    results = []

    methods = {
        "mean": {},
        "forward_fill": {},
        "backward_fill": {},
        "linear_interpolation": {},
        "spline_interpolation": {},
        "knn_temporal": {"n_neighbors": 5},
        "seasonal": {"period": 288},
        "rolling_mean": {"window": 10},
    }

    for method_name, params in methods.items():
        result = evaluate_imputation_method(series, method_name, **params)
        results.append(result)

        if result["success"]:
            status = "✓"
            msg = f"Missing: {result['missing_before']} → {result['missing_after']}, Time: {result['time_ms']:.2f}ms"
        else:
            status = "✗"
            msg = f"Error: {result['error'][:50]}"

        print(f"{status} {method_name:20} {msg}")

    # Add optional methods
    try:
        result = evaluate_imputation_method(series, "iterative", max_iter=10)
        results.append(result)
        if result["success"]:
            mb, ma, tm = result["missing_before"], result["missing_after"], result["time_ms"]
            print(f"✓ {'iterative':20} Missing: {mb} → {ma}, Time: {tm:.2f}ms")
        else:
            print(f"✗ {'iterative':20} {result['error'][:50]}")
    except Exception:
        print(f"⊘ {'iterative':20} Skipped (sklearn not available)")

    try:
        result = evaluate_imputation_method(series, "knn_sklearn", n_neighbors=5)
        results.append(result)
        if result["success"]:
            mb, ma, tm = result["missing_before"], result["missing_after"], result["time_ms"]
            print(f"✓ {'knn_sklearn':20} Missing: {mb} → {ma}, Time: {tm:.2f}ms")
        else:
            print(f"✗ {'knn_sklearn':20} {result['error'][:50]}")
    except Exception:
        print(f"⊘ {'knn_sklearn':20} Skipped (sklearn not available)")

    # Create DataFrame from successful results
    df_results = pd.DataFrame([r for r in results if r["success"]])

    if len(df_results) > 0:
        df_results = df_results.sort_values("time_ms")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Test imputation methods")
    parser.add_argument("--method", type=str, default=None, help="Test specific method")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV data file")
    parser.add_argument(
        "--synthetic", action="store_true", default=True, help="Use synthetic test data (default: True)"
    )
    parser.add_argument("--compare-all", action="store_true", default=False, help="Compare all methods")

    args = parser.parse_args()

    # Load data
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading data from {args.data_path}...")
        series = load_real_data(args.data_path)
        print(f"Loaded {len(series)} samples\n")
    else:
        print("Using synthetic test data...")
        series = create_synthetic_test_data()
        print(f"Created {len(series)} synthetic samples\n")

    print(f"Data: {series.isna().sum()} missing values ({series.isna().sum() / len(series) * 100:.1f}%)\n")

    # Test methods
    if args.compare_all:
        df_results = compare_all_methods(series)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(df_results.to_string(index=False))

    elif args.method:
        print(f"Testing method: {args.method}\n")
        result = evaluate_imputation_method(series, args.method)

        if result["success"]:
            print("✓ Success!")
            print(f"  Missing before: {result['missing_before']}")
            print(f"  Missing after: {result['missing_after']}")
            print(f"  Time: {result['time_ms']:.2f}ms")

            imputed = result["imputed_series"]
            print("\n  Statistics:")
            print(f"    Mean: {imputed.mean():.2f}")
            print(f"    Std: {imputed.std():.2f}")
            print(f"    Min: {imputed.min():.2f}")
            print(f"    Max: {imputed.max():.2f}")
        else:
            print(f"✗ Failed: {result['error']}")

    else:
        print("Available methods:")
        for method in sorted(imputation_methods.IMPUTATION_METHODS.keys()):
            print(f"  - {method}")
        print("\nUse --method <name> to test a specific method")
        print("Use --compare-all to compare all methods")


if __name__ == "__main__":
    main()
