"""
Zero-Shot Evaluation of CapnoBase-Trained Model on BIDMC Dataset

Evaluates a CorrEncoder model pre-trained on CapnoBase CO2 data against
the BIDMC RESP dataset without any fine-tuning.

This script demonstrates transfer learning evaluation:
- Source domain: CapnoBase (PPG → CO2)
- Target domain: BIDMC (PPG → RESP)
- Evaluation: Zero-shot (no model updates)

Key differences from LOSO evaluation:
- No training or fine-tuning
- Uses pre-trained capnobase_master.pth checkpoint
- Evaluates on all 53 BIDMC subjects
- Reconstructs continuous signals from pre-windowed data
- Uses sliding window inference with overlap-add (stride=30, 90% overlap)
- Evaluation metrics: Respiratory rate MAE on 30.6s and 60.6s windows

Author: Zhantao Wang
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.signal
import torch
from tqdm import tqdm

from utils import CorrEncoder, count_parameters
from loso_eval import calculate_respiratory_rate_fft, seed_everything, get_device, sliding_window_inference


# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint path
MODEL_CHECKPOINT = 'capnobase_master.pth'

# BIDMC data path
BIDMC_DATA_PATH = 'processed_data_bidmc/bidmc_all.pt'

# Results directory
RESULTS_DIR = 'bidmc_results'

# Dataset parameters
NUM_SUBJECTS = 53
WINDOWS_PER_SUBJECT = 50
WINDOW_SIZE = 288  # 9.6s at 30Hz
SAMPLING_RATE = 30  # Hz

# Sliding window inference parameters
STRIDE = 30  # 1 second at 30Hz (90% overlap with window_size=288)

# Inference batch size (for efficient GPU usage)
BATCH_SIZE = 32

# Reproducibility
RANDOM_SEED = 42

# Evaluation window sizes (matching paper specification)
# Paper uses 30.6s and 60.6s windows for respiratory rate calculation
# At 30Hz sampling rate:
WINDOW_SHORT_SEC = 30.6  # 30.6 seconds
SAMPLES_SHORT = 918      # 30.6s × 30Hz = 918 samples

WINDOW_LONG_SEC = 60.6   # 60.6 seconds
SAMPLES_LONG = 1818      # 60.6s × 30Hz = 1818 samples


# ============================================================================
# Data Loading
# ============================================================================

def load_bidmc_data(data_path):
    """
    Load BIDMC preprocessed data.

    Structure:
    - 'samples': [2650, 288] - PPG windows (normalized to [-1, 1])
    - 'labels': [2650, 288] - RESP windows (normalized to [0, 1])
    - Organization: 53 subjects × 50 windows
    - Subject i indices: [i*50 : (i+1)*50] for i in range(53)

    Args:
        data_path: Path to bidmc_all.pt

    Returns:
        data: Dict with 'samples' and 'labels' tensors
    """
    print("=" * 80)
    print("Loading BIDMC Dataset")
    print("=" * 80)

    data = torch.load(data_path)

    # Verify structure
    assert 'samples' in data and 'labels' in data, \
        "Data must contain 'samples' and 'labels' keys"

    samples = data['samples']  # [2650, 288]
    labels = data['labels']    # [2650, 288]

    assert samples.shape[0] == NUM_SUBJECTS * WINDOWS_PER_SUBJECT, \
        f"Expected {NUM_SUBJECTS * WINDOWS_PER_SUBJECT} windows, got {samples.shape[0]}"
    assert samples.shape[1] == WINDOW_SIZE, \
        f"Expected window size {WINDOW_SIZE}, got {samples.shape[1]}"

    print(f"Loaded BIDMC data successfully:")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Total subjects: {NUM_SUBJECTS}")
    print(f"  Windows per subject: {WINDOWS_PER_SUBJECT}")
    print(f"  Total windows: {samples.shape[0]}")
    print()

    return data


# ============================================================================
# Model Loading
# ============================================================================

def load_pretrained_model(checkpoint_path, device):
    """
    Load pre-trained CorrEncoder model from CapnoBase checkpoint.

    Checkpoint structure (from train_master_model.py):
    {
        'epoch': int,
        'model_state_dict': OrderedDict,
        'optimizer_state_dict': OrderedDict,
        'train_loss': float
    }

    Args:
        checkpoint_path: Path to capnobase_master.pth
        device: Device to load model on

    Returns:
        model: CorrEncoder model in eval mode
    """
    print("=" * 80)
    print("Loading Pre-trained Model")
    print("=" * 80)

    # Initialize model architecture
    model = CorrEncoder().to(device)

    print(f"Model architecture: CorrEncoder")
    print(f"Total parameters: {count_parameters(model):,}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode (disables dropout)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    print(f"  Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Model set to evaluation mode (dropout disabled)")
    print()

    return model


# ============================================================================
# Evaluation Logic
# ============================================================================

def compute_windowed_mae(pred, true, window_samples):
    """
    Calculate Mean Absolute Error for respiratory rate across non-overlapping windows.

    This function splits continuous signals into fixed-size windows, applies detrending,
    calculates respiratory rate via FFT for each window, and returns the mean error.

    Args:
        pred: Predicted continuous RESP signal [N_samples] (numpy array)
        true: Ground truth continuous RESP signal [N_samples] (numpy array)
        window_samples: Window size in samples (918 for 30.6s, 1818 for 60.6s)

    Returns:
        mae: Mean Absolute Error of respiratory rate in BPM across all windows
    """
    n_samples = len(pred)

    # Calculate number of complete non-overlapping windows
    num_windows = n_samples // window_samples

    # If signal is too short for even one window, return NaN
    if num_windows == 0:
        return float('nan')

    errors = []

    for i in range(num_windows):
        # Extract window
        start_idx = i * window_samples
        end_idx = start_idx + window_samples

        pred_window = pred[start_idx:end_idx]
        true_window = true[start_idx:end_idx]

        # CRITICAL: Apply linear detrending to remove baseline drift
        # This is essential for accurate FFT-based frequency estimation
        pred_detrended = scipy.signal.detrend(pred_window, type='linear')
        true_detrended = scipy.signal.detrend(true_window, type='linear')

        # Calculate respiratory rate via FFT
        rr_pred = calculate_respiratory_rate_fft(pred_detrended, fs=SAMPLING_RATE)
        rr_true = calculate_respiratory_rate_fft(true_detrended, fs=SAMPLING_RATE)

        # Compute absolute error for this window
        error = abs(rr_pred - rr_true)
        errors.append(error)

    # Return mean error across all windows
    return np.mean(errors)


def evaluate_subject(model, subject_idx, ppg_windows, resp_windows, device):
    """
    Evaluate model on a single BIDMC subject.

    Strategy (using sliding window inference):
    1. Flatten 50 pre-windowed PPG samples into continuous signal [14400]
    2. Apply sliding window inference with overlap-add (stride=30, ~471 windows)
    3. Flatten 50 ground truth RESP windows into continuous signal [14400]
    4. Calculate respiratory rate MAE for 30.6s windows (918 samples)
    5. Calculate respiratory rate MAE for 60.6s windows (1818 samples)

    Args:
        model: Pre-trained CorrEncoder in eval mode
        subject_idx: Subject index (0-52)
        ppg_windows: [50, 288] PPG windows for this subject
        resp_windows: [50, 288] RESP windows for this subject
        device: Device to run inference on

    Returns:
        Dict with evaluation metrics (subject_id, mae_30s, mae_60s)
    """
    # ========================================================================
    # Step 1: Reconstruct Continuous Signals from Pre-windowed Data
    # ========================================================================
    # Input: ppg_windows [50, 288], resp_windows [50, 288]
    # Output: continuous signals [14400] (50 * 288 samples = 480 seconds)

    ppg_continuous = ppg_windows.reshape(-1).numpy()  # [14400]
    resp_continuous = resp_windows.reshape(-1).numpy()  # [14400]

    # ========================================================================
    # Step 2: Sliding Window Inference with Overlap-Add
    # ========================================================================
    # Strategy: Create overlapping windows from continuous signal, run model
    # on each window, then merge predictions using weighted averaging.
    # This eliminates boundary artifacts that occur with simple concatenation.
    #
    # Parameters (matching LOSO evaluation):
    # - window_size: 288 samples (9.6s at 30Hz)
    # - stride: 30 samples (1s at 30Hz) → 89.6% overlap
    # - Result: ~471 overlapping windows from 14400 samples
    #
    # Benefits:
    # - Smooth continuous output (no stitching artifacts)
    # - Each sample averaged across ~9 predictions
    # - Matches methodology used in CapnoBase LOSO evaluation

    pred_continuous = sliding_window_inference(
        model=model,
        ppg_signal=ppg_continuous,
        device=device,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE
    )

    # ========================================================================
    # Step 3: Dual Metric Calculation (30.6s and 60.6s Windows)
    # ========================================================================
    # Paper methodology: Evaluate RR accuracy on two window sizes
    # - Short window (30.6s / 918 samples): Captures short-term RR variations
    # - Long window (60.6s / 1818 samples): Provides more stable RR estimates
    # Both windows use non-overlapping segmentation with linear detrending

    mae_30s = compute_windowed_mae(pred_continuous, resp_continuous, SAMPLES_SHORT)
    mae_60s = compute_windowed_mae(pred_continuous, resp_continuous, SAMPLES_LONG)

    # Format subject_id as zero-padded string: "01", "02", ..., "53"
    subject_id_str = f"{subject_idx + 1:02d}"

    return {
        'subject_id': subject_id_str,
        'mae_30s': mae_30s,
        'mae_60s': mae_60s
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def run_zeroshot_evaluation():
    """
    Main execution function for zero-shot BIDMC evaluation.

    Steps:
    1. Load BIDMC data (2650 windows)
    2. Load pre-trained CapnoBase model
    3. Loop through 53 subjects:
       - Extract subject's 50 windows
       - Run model inference
       - Flatten to continuous signal
       - Calculate RR error via FFT
    4. Save results to CSV
    5. Print summary statistics
    """
    # Set random seeds for reproducibility
    seed_everything(RANDOM_SEED)

    print("=" * 80)
    print("Zero-Shot Evaluation: CapnoBase Model on BIDMC Dataset")
    print("=" * 80)
    print()

    # Setup device
    device = get_device()
    print(f"Device: {device}")
    print()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    data = load_bidmc_data(BIDMC_DATA_PATH)
    samples = data['samples']  # [2650, 288]
    labels = data['labels']    # [2650, 288]

    # Load pre-trained model
    model = load_pretrained_model(MODEL_CHECKPOINT, device)

    # Storage for results
    all_results = []

    # Evaluation loop
    print("=" * 80)
    print("Evaluating All 53 BIDMC Subjects")
    print("=" * 80)
    print()

    start_time = time.time()

    for subject_idx in tqdm(range(NUM_SUBJECTS), desc="Subjects"):
        # Extract subject's windows
        start_idx = subject_idx * WINDOWS_PER_SUBJECT
        end_idx = start_idx + WINDOWS_PER_SUBJECT

        ppg_windows = samples[start_idx:end_idx]     # [50, 288]
        resp_windows = labels[start_idx:end_idx]     # [50, 288]

        # Verify shape
        assert ppg_windows.shape == (WINDOWS_PER_SUBJECT, WINDOW_SIZE), \
            f"Subject {subject_idx}: Expected shape ({WINDOWS_PER_SUBJECT}, {WINDOW_SIZE}), got {ppg_windows.shape}"

        # Evaluate subject
        result = evaluate_subject(model, subject_idx, ppg_windows, resp_windows, device)
        all_results.append(result)

    total_time = time.time() - start_time

    # Save results
    save_results(all_results)

    # Print summary
    print_summary(all_results, total_time)

    return all_results


# ============================================================================
# Results Management
# ============================================================================

def save_results(all_results):
    """
    Save evaluation results to CSV.

    Args:
        all_results: List of result dicts for 53 subjects
    """
    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'zeroshot_results.csv')
    df.to_csv(csv_path, index=False)

    print()
    print("=" * 80)
    print("Results Saved")
    print("=" * 80)
    print(f"CSV saved to: {csv_path}")
    print()


def print_summary(all_results, total_time):
    """
    Print summary statistics to console.

    Args:
        all_results: List of result dicts for 53 subjects
        total_time: Total evaluation time in seconds
    """
    df = pd.DataFrame(all_results)

    print()
    print("=" * 80)
    print("ZERO-SHOT EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Source Dataset:      CapnoBase (CO2)")
    print(f"Target Dataset:      BIDMC (RESP)")
    print(f"Model Checkpoint:    {MODEL_CHECKPOINT}")
    print(f"Subjects Evaluated:  {len(all_results)}")
    print(f"Evaluation Time:     {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print()
    print("Evaluation Window Statistics:")
    print()
    print("30.6-second Windows (918 samples):")
    print(f"  Mean MAE:    {df['mae_30s'].mean():.2f} BPM")
    print(f"  Median MAE:  {df['mae_30s'].median():.2f} BPM")
    print(f"  Std Dev:     {df['mae_30s'].std():.2f} BPM")
    print(f"  Min:         {df['mae_30s'].min():.2f} BPM")
    print(f"  Max:         {df['mae_30s'].max():.2f} BPM")
    print()
    print("60.6-second Windows (1818 samples):")
    print(f"  Mean MAE:    {df['mae_60s'].mean():.2f} BPM")
    print(f"  Median MAE:  {df['mae_60s'].median():.2f} BPM")
    print(f"  Std Dev:     {df['mae_60s'].std():.2f} BPM")
    print(f"  Min:         {df['mae_60s'].min():.2f} BPM")
    print(f"  Max:         {df['mae_60s'].max():.2f} BPM")
    print("=" * 80)
    print()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_zeroshot_evaluation()
        print("Zero-shot evaluation completed successfully!")

    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        print(f"Please ensure:")
        print(f"  1. {MODEL_CHECKPOINT} exists (run train_master_model.py)")
        print(f"  2. {BIDMC_DATA_PATH} exists (run preprocess_bidmc.py)")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(traceback.format_exc())
