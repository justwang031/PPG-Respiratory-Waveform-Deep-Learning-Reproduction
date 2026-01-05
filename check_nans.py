"""
Data Integrity Verification Script

Purpose: Verify whether NaN or Inf values exist in raw source files
         or if they are introduced during preprocessing.

This script scans all raw signal CSV files and checks for:
- NaN (Not a Number) values in PPG and Capnography columns
- Inf (Infinity) values in PPG and Capnography columns
"""

import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm


def check_file_integrity(filepath: Path) -> dict:
    """
    Check a single CSV file for NaN and Inf values.

    Args:
        filepath: Path to the CSV file to check

    Returns:
        dict with counts of NaNs and Infs found, or error information
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)

        # Verify required columns exist
        required_cols = ['pleth_y', 'co2_y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'error': f"Missing columns: {missing_cols}",
                'nan_count': 0,
                'inf_count': 0
            }

        # Check for NaN values in both columns
        nan_count = df[required_cols].isnull().sum().sum()

        # Check for Inf values in both columns
        inf_count = np.isinf(df[required_cols].values).sum()

        return {
            'error': None,
            'nan_count': int(nan_count),
            'inf_count': int(inf_count)
        }

    except Exception as e:
        return {
            'error': str(e),
            'nan_count': 0,
            'inf_count': 0
        }


def main():
    """
    Main execution: scan all raw signal files and report integrity status.
    """
    # Define the raw data directory and file pattern
    raw_data_dir = Path('raw_data')
    file_pattern = '*_signal.csv'

    # Check if directory exists
    if not raw_data_dir.exists():
        print(f"❌ ERROR: Directory '{raw_data_dir}' not found!")
        return

    # Find all matching files
    file_paths = sorted(raw_data_dir.glob(file_pattern))

    if not file_paths:
        print(f"⚠️  WARNING: No files matching pattern '{file_pattern}' found in '{raw_data_dir}'")
        return

    print(f"📊 Scanning {len(file_paths)} raw signal files for data integrity...\n")

    # Track issues found
    issues = {}
    total_nans = 0
    total_infs = 0

    # Process each file with progress bar
    for filepath in tqdm(file_paths, desc="Checking files", unit="file"):
        result = check_file_integrity(filepath)

        # Record any issues found
        if result['error']:
            issues[filepath.name] = f"Error: {result['error']}"
        elif result['nan_count'] > 0 or result['inf_count'] > 0:
            issue_parts = []
            if result['nan_count'] > 0:
                issue_parts.append(f"{result['nan_count']} NaNs")
                total_nans += result['nan_count']
            if result['inf_count'] > 0:
                issue_parts.append(f"{result['inf_count']} Infs")
                total_infs += result['inf_count']
            issues[filepath.name] = ", ".join(issue_parts)

    # Report results
    print("\n" + "="*70)
    if not issues:
        print("\033[1;32m✅ PASS: No NaNs or Infs found in raw data.\033[0m")
        print(f"All {len(file_paths)} files are clean!")
    else:
        print("\033[1;33m⚠️  FAIL: Found issues in the following files:\033[0m\n")
        for filename, issue_desc in issues.items():
            print(f"  • {filename}: {issue_desc}")
        print(f"\n📈 Summary:")
        print(f"  - Files with issues: {len(issues)}/{len(file_paths)}")
        print(f"  - Total NaN values: {total_nans}")
        print(f"  - Total Inf values: {total_infs}")
    print("="*70)


if __name__ == "__main__":
    main()
