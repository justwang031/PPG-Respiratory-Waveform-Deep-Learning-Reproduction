"""
PLS Baseline for PPG-to-CO2 Prediction using LOSO Cross-Validation

This script reproduces the Partial Least Squares (PLS) regression baseline
from the Deep Corr-Encoder paper. It uses the same data splits and evaluation
metrics as the Deep Learning model for fair comparison.

Paper specification: "PLS was implemented with 26 degrees of freedom"

Author: Zhantao Wang
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import gc
import json
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm

# Reuse existing evaluation infrastructure from loso_eval.py
from evaluation.loso_eval import (
    calculate_waveform_mae,
    calculate_rr_error_fft,
    calculate_duty_cycle_error,
    aggregate_loso_results,
    save_final_results,
    check_completed_folds,
    load_fold_results,
    save_fold_results,
    LOSODataLoader,
    seed_everything
)


# ============================================================================
# Configuration
# ============================================================================

# PLS Configuration (from paper: "PLS was implemented with 26 degrees of freedom")
N_COMPONENTS = 26

# Windowing parameters (must match Deep Learning baseline)
WINDOW_SIZE = 288  # 9.6s at 30Hz
STRIDE = 30        # 1 second at 30Hz for sliding window inference

# Results directory
RESULTS_DIR = 'pls_results'

# Reproducibility seed
RANDOM_SEED = 42


# ============================================================================
# Core Functions
# ============================================================================

def numpy_sliding_window_inference(pls_model, ppg_signal, window_size=288, stride=30):
    """
    Perform sliding window inference using PLS model with overlap-add averaging.

    This replicates the exact reconstruction logic from sliding_window_inference()
    in loso_eval.py, but for scikit-learn PLS instead of PyTorch CorrEncoder.

    The reconstruction uses overlap-add averaging:
    1. Create overlapping windows (stride < window_size)
    2. Predict each window independently
    3. Accumulate predictions in overlapping regions
    4. Average by dividing by overlap counts

    Args:
        pls_model: Fitted PLSRegression model
        ppg_signal: Continuous PPG signal [N_samples] - 1D numpy array
        window_size: 288 samples (9.6s at 30Hz)
        stride: 30 samples (1 second at 30Hz)

    Returns:
        fused_prediction: [N_samples] continuous CO2 prediction
    """
    n_samples = len(ppg_signal)

    # STEP 1: Create sliding windows
    # Same logic as create_sliding_windows() from loso_eval.py
    windows = []
    start_indices = []

    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        windows.append(ppg_signal[start_idx:end_idx])
        start_indices.append(start_idx)

    windows = np.array(windows)  # Shape: [N_windows, 288]

    # STEP 2: Predict on all windows
    # PLS expects [N_samples, N_features], windows is already correct shape
    predictions = pls_model.predict(windows)  # Shape: [N_windows, 288]

    # STEP 3: Initialize accumulation arrays (exact same as PyTorch version)
    accumulated_signal = np.zeros(n_samples, dtype=np.float32)
    count_array = np.zeros(n_samples, dtype=np.float32)

    # STEP 4: Accumulate overlapping predictions with same logic as loso_eval.py
    for window_idx, start_idx in enumerate(start_indices):
        end_idx = start_idx + window_size

        # Handle edge case: last window might extend beyond signal
        if end_idx > n_samples:
            valid_length = n_samples - start_idx
            accumulated_signal[start_idx:] += predictions[window_idx, :valid_length]
            count_array[start_idx:] += 1
        else:
            accumulated_signal[start_idx:end_idx] += predictions[window_idx]
            count_array[start_idx:end_idx] += 1

    # STEP 5: Average where overlaps exist (exact same divide logic)
    fused_prediction = np.divide(
        accumulated_signal,
        count_array,
        where=count_array > 0,
        out=np.zeros_like(accumulated_signal)
    )

    return fused_prediction


def collect_training_data(train_loader):
    """
    Collect all training data from PyTorch DataLoader into NumPy arrays.

    Converts PyTorch tensors [batch_size, 1, 288] to NumPy arrays [N_total, 288]
    for use with scikit-learn PLSRegression.

    Args:
        train_loader: PyTorch DataLoader with batches of shape [batch_size, 1, 288]

    Returns:
        X_train: [N_total_samples, 288] - PPG windows
        y_train: [N_total_samples, 288] - CO2 windows
    """
    X_list = []
    y_list = []

    # Iterate through all batches
    for ppg_batch, co2_batch in train_loader:
        # ppg_batch: [batch_size, 1, 288]
        # co2_batch: [batch_size, 1, 288]

        # Convert to numpy and remove channel dimension
        ppg_np = ppg_batch.squeeze(1).numpy()  # [batch_size, 288]
        co2_np = co2_batch.squeeze(1).numpy()  # [batch_size, 288]

        X_list.append(ppg_np)
        y_list.append(co2_np)

    # Concatenate all batches
    X_train = np.concatenate(X_list, axis=0)  # [N_total, 288]
    y_train = np.concatenate(y_list, axis=0)  # [N_total, 288]

    return X_train, y_train


# ============================================================================
# Main LOSO Cross-Validation
# ============================================================================

def run_pls_loso_cross_validation():
    """
    Main execution loop for 42-fold PLS LOSO cross-validation.

    For each fold:
    1. Get LOSO split from LOSODataLoader
    2. Collect training data (flatten PyTorch DataLoader to NumPy)
    3. Fit PLS model with n_components=26
    4. Run sliding window inference on test subject
    5. Calculate 3 metrics (MAE, RR error, duty cycle error)
    6. Save fold results

    After all folds:
    7. Aggregate results (MEDIAN across folds)
    8. Save CSV and JSON outputs
    """
    print("=" * 80)
    print("PLS Baseline LOSO Cross-Validation")
    print(f"N_COMPONENTS = {N_COMPONENTS}")
    print(f"WINDOW_SIZE = {WINDOW_SIZE}")
    print(f"STRIDE = {STRIDE}")
    print("=" * 80)
    print()

    # Set random seeds for reproducibility
    seed_everything(RANDOM_SEED)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize data loader (reuse from loso_eval.py)
    print("Initializing LOSODataLoader...")
    data_loader = LOSODataLoader()
    print(f"Total subjects: {len(data_loader.subject_ids)}")
    print()

    # Check for completed folds (resume support)
    completed_folds = check_completed_folds(RESULTS_DIR)
    if completed_folds:
        print(f"Found {len(completed_folds)} completed folds: {sorted(completed_folds)}")
        print("Will skip these and resume from next fold.")
        print()

    all_results = []

    # ========================================================================
    # LOSO LOOP (42 folds)
    # ========================================================================

    for fold_idx in tqdm(range(42), desc="PLS LOSO Folds"):
        # Skip if already completed
        if fold_idx in completed_folds:
            fold_results = load_fold_results(fold_idx, RESULTS_DIR)
            all_results.append(fold_results)
            continue

        fold_start_time = time.time()
        fold_dir = os.path.join(RESULTS_DIR, f'fold_{fold_idx:02d}')

        try:
            print(f"\n{'='*80}")
            print(f"Fold {fold_idx + 1}/42")
            print(f"{'='*80}")

            # Reset seed for reproducibility
            seed_everything(RANDOM_SEED)

            # Get LOSO split (reuse LOSODataLoader)
            train_loader, test_data = data_loader.get_loso_split(fold_idx)
            print(f"Test subject: {test_data['subject_id']}")
            print(f"Training windows: {len(train_loader.dataset)}")
            print(f"Test signal length: {len(test_data['ppg_continuous'])} samples")

            # Collect training data from DataLoader
            print("Collecting training data...")
            X_train, y_train = collect_training_data(train_loader)
            print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

            # Initialize and fit PLS model
            print(f"Fitting PLS model (n_components={N_COMPONENTS})...")
            # scale=False because data is already normalized (PPG: [-1,1], CO2: [0,1])
            pls_model = PLSRegression(n_components=N_COMPONENTS, scale=False)
            pls_model.fit(X_train, y_train)
            print("PLS model fitted successfully")

            # Run sliding window inference on test data
            print(f"Evaluating with sliding window inference (stride={STRIDE})...")
            ppg_continuous = test_data['ppg_continuous']
            co2_continuous = test_data['co2_continuous']

            # Use numpy_sliding_window_inference()
            fused_prediction = numpy_sliding_window_inference(
                pls_model, ppg_continuous,
                window_size=WINDOW_SIZE, stride=STRIDE
            )

            # Calculate all three metrics
            waveform_mae = calculate_waveform_mae(fused_prediction, co2_continuous)
            rr_error = calculate_rr_error_fft(fused_prediction, co2_continuous)
            duty_cycle_error = calculate_duty_cycle_error(fused_prediction, co2_continuous)

            # Save fold results
            fold_time = time.time() - fold_start_time
            fold_results = {
                'fold': fold_idx,
                'subject_id': test_data['subject_id'],
                'waveform_mae': waveform_mae,
                'rr_error_bpm': rr_error,
                'duty_cycle_error': duty_cycle_error,
                'n_components': N_COMPONENTS,
                'fold_time_seconds': fold_time
            }

            # Create fold directory and save results
            os.makedirs(fold_dir, exist_ok=True)
            save_fold_results(fold_results, fold_dir)
            all_results.append(fold_results)

            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Waveform MAE:        {waveform_mae:.6f}")
            print(f"  RR Error (BPM):      {rr_error:.2f}")
            print(f"  Duty Cycle Error:    {duty_cycle_error:.4f}")
            print(f"  Time:                {fold_time:.2f} seconds")

            # Memory cleanup
            del pls_model, train_loader, test_data, X_train, y_train
            gc.collect()

        except Exception as e:
            # Error handling - log but continue
            error_msg = f"Fold {fold_idx} failed: {str(e)}"
            print(f"\nERROR: {error_msg}")
            print("Full traceback:")
            print(traceback.format_exc())

            # Save error log to fold directory
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, 'error.log'), 'w') as f:
                f.write(f"Fold {fold_idx} Error Log\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\n{error_msg}\n\n")
                f.write("Full traceback:\n")
                f.write(traceback.format_exc())

            print(f"Error log saved to: {os.path.join(fold_dir, 'error.log')}")
            print("Continuing to next fold...\n")
            continue

    # ========================================================================
    # AGGREGATION PHASE
    # ========================================================================

    print(f"\n{'='*80}")
    print("Aggregating Results")
    print(f"{'='*80}")

    if not all_results:
        print("ERROR: No successful folds to aggregate!")
        return None

    # Aggregate results (reuse from loso_eval.py)
    aggregated = aggregate_loso_results(all_results)

    # Modify method name for PLS baseline
    final_results = {
        'method': 'PLS Baseline LOSO Cross-Validation',
        'n_components': N_COMPONENTS,
        'n_folds': len(all_results),
        'aggregation': 'median',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': aggregated
    }

    # Save final outputs
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, 'pls_all_fold_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nAll fold results saved to: {csv_path}")

    # Save aggregated JSON
    json_path = os.path.join(RESULTS_DIR, 'aggregated_results.json')
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Aggregated results saved to: {json_path}")

    # Print final results
    print(f"\n{'='*80}")
    print(f"PLS BASELINE FINAL RESULTS (MEDIAN across {len(all_results)} folds)")
    print(f"N_COMPONENTS = {N_COMPONENTS}")
    print(f"{'='*80}")
    print(f"Waveform MAE:       {aggregated['waveform_mae']['median']:.6f} "
          f"(±{aggregated['waveform_mae']['std']:.6f})")
    print(f"RR Error (BPM):     {aggregated['rr_error_bpm']['median']:.2f} "
          f"(±{aggregated['rr_error_bpm']['std']:.2f})")
    print(f"Duty Cycle Error:   {aggregated['duty_cycle_error']['median']:.4f} "
          f"(±{aggregated['duty_cycle_error']['std']:.4f})")
    print(f"{'='*80}")

    return aggregated


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        results = run_pls_loso_cross_validation()

        if results is not None:
            print("\nPLS baseline evaluation completed successfully!")
            print(f"Results saved to: {RESULTS_DIR}/")
        else:
            print("\nPLS baseline evaluation failed - no results generated.")

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        print("Partial results may be saved in individual fold directories.")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
