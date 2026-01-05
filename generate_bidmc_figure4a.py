"""
Figure 4(a) Generator for BIDMC Dataset - Window Length Sweep Analysis

Reproduces Figure 4(a) from the paper comparing respiratory rate estimation errors
across window lengths (10s to 120s) for two training scenarios:

- Train C (Zero-shot): Pre-trained on CapnoBase, tested on BIDMC without fine-tuning
- Train C+B (Fine-tuning): Pre-trained on CapnoBase, fine-tuned on BIDMC via LOSO

Each scenario displays two metrics:
- Solid line: Median of ALL individual window errors across ALL subjects
- Dashed line: mMAE (median of per-subject mean errors)

Author: Zhantao Wang
"""

import os
import gc
import pickle
from datetime import datetime

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from utils import CorrEncoder
from loso_eval import (
    seed_everything,
    get_device,
    sliding_window_inference
)


# ============================================================================
# Configuration
# ============================================================================

# Dataset parameters
NUM_SUBJECTS = 53
WINDOWS_PER_SUBJECT = 50
WINDOW_SIZE = 288  # 9.6s at 30Hz
SAMPLING_RATE = 30  # Hz

# Sliding window inference
STRIDE = 30  # 1s overlap (90% overlap with window_size=288)
BATCH_SIZE = 32

# Fine-tuning hyperparameters
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# Window sweep configuration
WINDOW_SIZES = list(range(10, 121, 2))  # [10, 12, 14, ..., 120] seconds

# Reproducibility
RANDOM_SEED = 42

# Figure settings
FIGURE_DPI = 300

# File paths
BIDMC_DATA_PATH = 'processed_data_bidmc/bidmc_all.pt'
MODEL_CHECKPOINT = 'capnobase_master.pth'
OUTPUT_DIR_PLOTS = 'bidmc_plots'
OUTPUT_DIR_DATA = 'paper_plots'


# ============================================================================
# DSP Helper Functions: Decoupled RR Estimators
# ============================================================================

def bandpass_filter(signal, fs, low=0.1, high=0.7, order=4):
    """
    Apply Butterworth bandpass filter to isolate respiratory frequencies.

    Args:
        signal: Input signal (numpy array)
        fs: Sampling rate (Hz)
        low: Lower cutoff (Hz, default 0.1 = 6 BPM)
        high: Upper cutoff (Hz, default 0.7 = 42 BPM)
        order: Filter order (default 4)

    Returns:
        Filtered signal (zero-phase using filtfilt)
    """
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq

    sos = scipy.signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
    filtered = scipy.signal.sosfiltfilt(sos, signal)

    return filtered


def calculate_respiratory_rate_fft_precise(signal, fs):
    """
    Calculate respiratory rate using FFT with sub-bin parabolic interpolation.

    This method provides higher frequency resolution than simple peak-picking
    by interpolating between FFT bins using parabolic peak fitting.

    Args:
        signal: Respiratory signal (numpy array)
        fs: Sampling rate (Hz)

    Returns:
        Respiratory rate in BPM (float)
    """
    # Step 1: Remove DC component
    signal_centered = signal - np.mean(signal)

    # Step 2: Apply Hann window to reduce spectral leakage
    window = scipy.signal.windows.hann(len(signal_centered))
    signal_windowed = signal_centered * window

    # Step 3: Compute FFT
    fft_vals = np.fft.rfft(signal_windowed)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(signal_windowed), d=1/fs)

    # Step 4: Restrict to physiological range (0.1-0.7 Hz)
    mask = (freqs >= 0.1) & (freqs <= 0.7)
    valid_fft = fft_mag[mask]
    valid_freqs = freqs[mask]

    if len(valid_fft) == 0:
        return 0.0  # Fallback

    # Step 5: Find peak bin
    peak_idx = np.argmax(valid_fft)

    # Step 6: Parabolic interpolation for sub-bin accuracy
    # Use magnitudes at k-1, k, k+1
    if peak_idx > 0 and peak_idx < len(valid_fft) - 1:
        alpha = valid_fft[peak_idx - 1]
        beta = valid_fft[peak_idx]
        gamma = valid_fft[peak_idx + 1]

        # Parabolic peak position (fractional bin offset)
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)

        # Interpolated frequency
        peak_freq = valid_freqs[peak_idx] + p * (valid_freqs[1] - valid_freqs[0])
    else:
        # Edge case: peak at boundary, no interpolation
        peak_freq = valid_freqs[peak_idx]

    # Step 7: Convert to BPM
    rr_bpm = peak_freq * 60

    return rr_bpm


def calculate_respiratory_rate_peaks(signal, fs):
    """
    Calculate respiratory rate using time-domain peak counting.

    This method mimics manual peak counting and avoids FFT grid quantization.
    Uses median peak-to-peak interval for robustness.

    Args:
        signal: Respiratory signal (numpy array)
        fs: Sampling rate (Hz)

    Returns:
        Respiratory rate in BPM (float)
    """
    # Step 1: Bandpass filter (0.1-0.7 Hz)
    filtered = bandpass_filter(signal, fs, low=0.1, high=0.7, order=4)

    # Step 2: Detect peaks with quality constraints
    min_distance = int(fs * 60 / 45)  # Minimum distance for 45 BPM max

    # Dynamic prominence threshold (guard against flat signals)
    ptp = np.ptp(filtered)
    if ptp < 1e-6:
        # Fallback to FFT for flat signals
        return calculate_respiratory_rate_fft_precise(signal, fs)

    prominence = 0.2 * ptp

    peaks, _ = scipy.signal.find_peaks(
        filtered,
        distance=min_distance,
        prominence=prominence
    )

    # Step 3: Quality check - need at least 2 peaks
    if len(peaks) < 2:
        # Fallback to FFT
        return calculate_respiratory_rate_fft_precise(signal, fs)

    # Step 4: Calculate inter-peak intervals
    dt = np.diff(peaks) / fs  # Time differences in seconds

    # Step 5: Robust RR using median interval
    median_interval = np.median(dt)
    rr = 60.0 / median_interval

    # Step 6: Quality check - RR in valid range and consistent intervals
    std_dt = np.std(dt)
    mean_dt = np.mean(dt)
    cv = std_dt / mean_dt if mean_dt > 0 else 999  # Coefficient of variation

    if rr < 6 or rr > 42 or cv > 0.3:
        # Fallback to FFT if estimate is unreliable
        return calculate_respiratory_rate_fft_precise(signal, fs)

    return rr


# ============================================================================
# Core Helper Function: Window Sweep Metrics
# ============================================================================

def calculate_sweep_metrics(pred_continuous, gt_continuous, window_sizes, fs=30):
    """
    Calculate RR absolute errors across multiple window sizes for one subject.

    This function splits the continuous predicted and ground-truth respiratory signals
    into non-overlapping windows of varying durations. For each window, it removes
    baseline drift via linear detrending, estimates respiratory rate (RR) using
    *decoupled* estimators, and stores the absolute RR error:

    - rr_pred: Frequency-domain FFT peak with Hann windowing and sub-bin parabolic
      interpolation (continuous-valued RR estimate).
    - rr_true: Time-domain peak counting on the detrended ground-truth signal after
      bandpass filtering (0.1–0.7 Hz), using a robust median inter-peak interval.
      If peak-based estimation is unreliable (e.g., too few peaks, RR out of range,
      highly variable intervals), it falls back to the FFT-precise estimator.

    Decoupling the RR estimators for prediction and ground truth avoids the shared
    FFT-bin quantization effect that can create staircase-like artifacts in median
    error curves.

    Args:
        pred_continuous: Predicted continuous RESP signal [N_samples] (numpy array).
        gt_continuous: Ground-truth continuous RESP signal [N_samples] (numpy array).
        window_sizes: List of window durations in seconds (e.g., [10, 12, 14, ...]).
        fs: Sampling rate in Hz (default: 30).

    Returns:
        Dict mapping window_size (seconds) -> list of absolute RR errors (BPM) for
        all windows of that size.
        Example: {10: [1.2, 0.8, 1.5], 12: [0.9, 1.1], ...}
    """
    results = {}

    for window_sec in window_sizes:
        window_samples = window_sec * fs  # Convert seconds to samples
        num_windows = len(pred_continuous) // window_samples

        # If signal is too short for even one window, return empty list
        if num_windows == 0:
            results[window_sec] = []
            continue

        errors = []
        for i in range(num_windows):
            # Extract non-overlapping chunk
            start = i * window_samples
            end = start + window_samples

            pred_chunk = pred_continuous[start:end]
            gt_chunk = gt_continuous[start:end]

            # CRITICAL: Apply linear detrending to remove baseline drift
            pred_detrended = scipy.signal.detrend(pred_chunk, type='linear')
            gt_detrended = scipy.signal.detrend(gt_chunk, type='linear')

            # Calculate respiratory rates using DECOUPLED estimators:
            # - Predictions: FFT with sub-bin parabolic interpolation (continuous values)
            # - Ground truth: Time-domain peak counting (mimics manual counting)
            # This eliminates the quantization staircase artifact by ensuring pred and
            # truth are not constrained to the same discrete frequency grid.

            rr_pred = calculate_respiratory_rate_fft_precise(pred_detrended, fs)
            rr_true = calculate_respiratory_rate_peaks(gt_detrended, fs)

            # Store absolute error in BPM
            errors.append(abs(rr_pred - rr_true))

        results[window_sec] = errors

    return results


# ============================================================================
# Zero-Shot Evaluation Pipeline
# ============================================================================

def run_zeroshot_sweep(device):
    """
    Evaluate pre-trained CapnoBase model on all 53 BIDMC subjects (zero-shot).

    This function:
    1. Loads BIDMC data and pre-trained model once
    2. For each subject:
       - Reconstructs continuous signals from pre-windowed data
       - Runs sliding window inference with overlap-add
       - Calculates RR errors across all window sizes (10s-120s)
    3. Returns nested error structure preserving subject boundaries

    Args:
        device: torch.device to run inference on

    Returns:
        Dict mapping window_size → list of per-subject error lists
        Structure: {window_size: [[subj0_errors], [subj1_errors], ...]}
    """
    print("Loading BIDMC data...")
    data = torch.load(BIDMC_DATA_PATH)
    samples = data['samples']  # [2650, 288]
    labels = data['labels']    # [2650, 288]

    print("Loading pre-trained model...")
    model = CorrEncoder().to(device)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from: {MODEL_CHECKPOINT}")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    print(f"  Training loss: {checkpoint['train_loss']:.6f}\n")

    # Initialize nested structure to preserve subject boundaries
    all_errors = {w: [] for w in WINDOW_SIZES}

    print("Evaluating all 53 BIDMC subjects (zero-shot)...")
    for subject_idx in tqdm(range(NUM_SUBJECTS), desc="Zero-shot subjects"):
        # Extract subject's 50 windows
        start_idx = subject_idx * WINDOWS_PER_SUBJECT
        end_idx = start_idx + WINDOWS_PER_SUBJECT

        ppg_windows = samples[start_idx:end_idx]     # [50, 288]
        resp_windows = labels[start_idx:end_idx]     # [50, 288]

        # Flatten to continuous signal [14400 samples = 480 seconds]
        ppg_continuous = ppg_windows.reshape(-1).numpy()
        resp_continuous = resp_windows.reshape(-1).numpy()

        # Sliding window inference with overlap-add (stride=30, 90% overlap)
        pred_continuous = sliding_window_inference(
            model=model,
            ppg_signal=ppg_continuous,
            device=device,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE
        )

        # Calculate errors across all window sizes for this subject
        subject_errors = calculate_sweep_metrics(
            pred_continuous, resp_continuous, WINDOW_SIZES, fs=SAMPLING_RATE
        )

        # Store as separate list to preserve subject boundary
        for w in WINDOW_SIZES:
            all_errors[w].append(subject_errors[w])

    return all_errors


# ============================================================================
# Fine-Tuning LOSO Evaluation Pipeline
# ============================================================================

def run_finetune_loso_sweep(device):
    """
    Perform 53-fold LOSO cross-validation with fine-tuning and window sweep.

    This function:
    1. For each of 53 LOSO folds:
       - Splits data: 52 subjects for training, 1 for testing
       - Initializes fresh model with pre-trained CapnoBase weights
       - Fine-tunes for 15 epochs with LR=1e-4
       - Evaluates on held-out subject with sliding window inference
       - Calculates RR errors across all window sizes
       - Cleans up memory to prevent OOM
    2. Returns nested error structure preserving subject boundaries

    Args:
        device: torch.device to run training/inference on

    Returns:
        Dict mapping window_size → list of per-subject error lists
        Structure: {window_size: [[subj0_errors], [subj1_errors], ...]}
    """
    print("Loading BIDMC data...")
    data = torch.load(BIDMC_DATA_PATH)
    samples = data['samples']  # [2650, 288]
    labels = data['labels']    # [2650, 288]

    # Initialize nested structure
    all_errors = {w: [] for w in WINDOW_SIZES}

    print("Starting 53-fold LOSO fine-tuning...")
    print(f"Fine-tuning config: LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}\n")

    for subject_idx in tqdm(range(NUM_SUBJECTS), desc="LOSO folds"):
        # ====================================================================
        # STEP 1: LOSO Data Split
        # ====================================================================
        test_start = subject_idx * WINDOWS_PER_SUBJECT
        test_end = (subject_idx + 1) * WINDOWS_PER_SUBJECT

        test_ppg = samples[test_start:test_end]      # [50, 288]
        test_resp = labels[test_start:test_end]      # [50, 288]

        # Training data: all subjects except current one
        if subject_idx == 0:
            # First subject: train on subjects 1-52
            train_ppg = samples[test_end:]
            train_resp = labels[test_end:]
        elif subject_idx == NUM_SUBJECTS - 1:
            # Last subject: train on subjects 0-51
            train_ppg = samples[:test_start]
            train_resp = labels[:test_start]
        else:
            # Middle subject: concatenate before and after
            train_ppg = torch.cat([samples[:test_start], samples[test_end:]])
            train_resp = torch.cat([labels[:test_start], labels[test_end:]])

        # ====================================================================
        # STEP 2: Create DataLoader
        # ====================================================================
        train_dataset = TensorDataset(
            train_ppg.unsqueeze(1),   # Add channel dimension: [N, 288] → [N, 1, 288]
            train_resp.unsqueeze(1)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # ====================================================================
        # STEP 3: Initialize Model with Pre-trained Weights
        # ====================================================================
        model = CorrEncoder().to(device)

        # CRITICAL: Load pre-trained weights from CapnoBase
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # ====================================================================
        # STEP 4: Fine-tune for 15 epochs
        # ====================================================================
        model.train()
        for epoch in range(NUM_EPOCHS):
            for ppg_batch, resp_batch in train_loader:
                ppg_batch = ppg_batch.to(device)
                resp_batch = resp_batch.to(device)

                # Forward pass
                pred = model(ppg_batch)
                loss = criterion(pred, resp_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ====================================================================
        # STEP 5: Evaluate on Test Subject
        # ====================================================================
        model.eval()

        # Flatten to continuous signal
        ppg_continuous = test_ppg.reshape(-1).numpy()
        resp_continuous = test_resp.reshape(-1).numpy()

        # Sliding window inference
        pred_continuous = sliding_window_inference(
            model=model,
            ppg_signal=ppg_continuous,
            device=device,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE
        )

        # Calculate errors across all window sizes
        subject_errors = calculate_sweep_metrics(
            pred_continuous, resp_continuous, WINDOW_SIZES, fs=SAMPLING_RATE
        )

        # Store with subject boundary preserved
        for w in WINDOW_SIZES:
            all_errors[w].append(subject_errors[w])

        # ====================================================================
        # STEP 6: Memory Cleanup
        # ====================================================================
        del model, optimizer, train_loader, train_dataset, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return all_errors


# ============================================================================
# Metric Aggregation
# ============================================================================

def aggregate_metrics(all_errors):
    """
    Compute median and mMAE from nested error structure.

    This function processes the nested error structure (preserving subject boundaries)
    to compute two key metrics:
    1. Median of ALL individual window errors across all subjects
    2. mMAE: Median of per-subject mean absolute errors

    Args:
        all_errors: Dict mapping window_size → list of per-subject error lists
                   Structure: {window_size: [[subj0_errors], [subj1_errors], ...]}

    Returns:
        Dict with keys:
            'window_sizes': List of window sizes in seconds
            'median_all': List of median values across ALL errors
            'median_mae': List of mMAE values (median of per-subject MAEs)
    """
    window_sizes = sorted(all_errors.keys())
    median_all = []
    median_mae = []

    for w in window_sizes:
        subject_error_lists = all_errors[w]  # [[subj0], [subj1], ...]

        # Metric 1: Median of ALL individual window errors
        all_individual_errors = [
            error
            for subj_errors in subject_error_lists
            for error in subj_errors
        ]

        if len(all_individual_errors) > 0:
            median_all.append(np.median(all_individual_errors))
        else:
            median_all.append(np.nan)

        # Metric 2: mMAE - Median of per-subject MAEs
        per_subject_maes = [
            np.mean(subj_errors)
            for subj_errors in subject_error_lists
            if len(subj_errors) > 0  # Skip empty lists
        ]

        if len(per_subject_maes) > 0:
            median_mae.append(np.median(per_subject_maes))
        else:
            median_mae.append(np.nan)

    return {
        'window_sizes': window_sizes,
        'median_all': median_all,
        'median_mae': median_mae
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_figure_4a(plot_data, output_path):
    """
    Generate Figure 4(a) with red/purple solid/dashed lines.

    Creates a publication-quality plot comparing respiratory rate estimation errors
    across window lengths for zero-shot (Train C) and fine-tuning (Train C+B) scenarios.

    Args:
        plot_data: Dict with structure:
            {
                'Train C': {'window_sizes': [...], 'median_all': [...], 'median_mae': [...]},
                'Train C+B': {'window_sizes': [...], 'median_all': [...], 'median_mae': [...]}
            }
        output_path: Path to save the figure (PNG format)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Zero-shot (Train C) - Red lines
    ax.plot(
        plot_data['Train C']['window_sizes'],
        plot_data['Train C']['median_all'],
        color='red', linestyle='-', linewidth=2,
        label='Train C: Median'
    )

    ax.plot(
        plot_data['Train C']['window_sizes'],
        plot_data['Train C']['median_mae'],
        color='red', linestyle='--', linewidth=2,
        label='Train C: mMAE'
    )

    # Fine-tuning (Train C+B) - Purple lines
    ax.plot(
        plot_data['Train C+B']['window_sizes'],
        plot_data['Train C+B']['median_all'],
        color='purple', linestyle='-', linewidth=2,
        label='Train C+B: Median'
    )

    ax.plot(
        plot_data['Train C+B']['window_sizes'],
        plot_data['Train C+B']['median_mae'],
        color='purple', linestyle='--', linewidth=2,
        label='Train C+B: mMAE'
    )

    # Formatting
    ax.set_xlabel('Window length (s)', fontsize=12)
    ax.set_ylabel('Respiratory rate error (Breaths/min)', fontsize=12)
    ax.set_title('Respiratory rate estimation: BIDMC', fontsize=14, fontweight='bold')
    ax.set_xlim(10, 120)
    ax.set_ylim(0, 5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"Figure saved: {output_path}")


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    """
    Main execution function orchestrating the entire pipeline.

    Steps:
    1. Setup: Seed random generators, detect device, create directories
    2. Phase 1: Zero-shot evaluation (~5 minutes)
    3. Phase 2: Fine-tuning LOSO evaluation (2-4 hours)
    4. Phase 3: Aggregate metrics and generate figure
    5. Save results to pickle and PNG files
    """
    try:
        # ====================================================================
        # Setup
        # ====================================================================
        print("=" * 80)
        print("BIDMC Figure 4(a) Generator: Window Length Sweep Analysis")
        print("=" * 80)
        print()

        # Check input files exist
        if not os.path.exists(BIDMC_DATA_PATH):
            raise FileNotFoundError(
                f"BIDMC data not found: {BIDMC_DATA_PATH}\n"
                f"Please run preprocess_bidmc.py first."
            )

        if not os.path.exists(MODEL_CHECKPOINT):
            raise FileNotFoundError(
                f"Pre-trained model not found: {MODEL_CHECKPOINT}\n"
                f"Please run train_master_model.py first."
            )

        # Set random seed for reproducibility
        seed_everything(RANDOM_SEED)

        # Detect device
        device = get_device()
        print(f"Device: {device}")
        print(f"Random seed: {RANDOM_SEED}")
        print(f"Window sweep range: {WINDOW_SIZES[0]}s to {WINDOW_SIZES[-1]}s (step {WINDOW_SIZES[1] - WINDOW_SIZES[0]}s)")
        print(f"Total window sizes: {len(WINDOW_SIZES)}")
        print()

        # Create output directories
        os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
        os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)

        # ====================================================================
        # Phase 1: Zero-Shot Evaluation
        # ====================================================================
        print("=" * 80)
        print("PHASE 1: Zero-Shot Evaluation (Train C)")
        print("=" * 80)
        print("Strategy: Pre-trained on CapnoBase, tested on BIDMC")
        print()

        zeroshot_errors = run_zeroshot_sweep(device)

        print(f"\nZero-shot evaluation complete!")
        print(f"  Subjects evaluated: {NUM_SUBJECTS}")
        print(f"  Window sizes tested: {len(WINDOW_SIZES)}")
        print()

        # ====================================================================
        # Phase 2: Fine-Tuning LOSO Evaluation
        # ====================================================================
        print("=" * 80)
        print("PHASE 2: Fine-Tuning LOSO Evaluation (Train C+B)")
        print("=" * 80)
        print("Strategy: Pre-trained on CapnoBase, fine-tuned on BIDMC via LOSO")
        print()

        finetune_errors = run_finetune_loso_sweep(device)

        print(f"\nFine-tuning LOSO evaluation complete!")
        print(f"  LOSO folds completed: {NUM_SUBJECTS}")
        print(f"  Window sizes tested: {len(WINDOW_SIZES)}")
        print()

        # ====================================================================
        # Phase 3: Aggregation & Visualization
        # ====================================================================
        print("=" * 80)
        print("PHASE 3: Metric Aggregation & Visualization")
        print("=" * 80)
        print()

        print("Aggregating metrics...")
        plot_data = {
            'Train C': aggregate_metrics(zeroshot_errors),
            'Train C+B': aggregate_metrics(finetune_errors),
            'metadata': {
                'sampling_rate': SAMPLING_RATE,
                'num_subjects': NUM_SUBJECTS,
                'window_step': WINDOW_SIZES[1] - WINDOW_SIZES[0],
                'num_window_sizes': len(WINDOW_SIZES),
                'timestamp': datetime.now().isoformat(),
                'fine_tuning_lr': LEARNING_RATE,
                'fine_tuning_epochs': NUM_EPOCHS
            }
        }

        # Save pickle for reproducibility
        pickle_path = os.path.join(OUTPUT_DIR_DATA, 'plot_data_bidmc.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"Saved aggregated data: {pickle_path}")

        # Generate figure
        figure_path = os.path.join(OUTPUT_DIR_PLOTS, 'Figure_4a_BIDMC.png')
        print("\nGenerating Figure 4(a)...")
        plot_figure_4a(plot_data, figure_path)

        # ====================================================================
        # Summary
        # ====================================================================
        print()
        print("=" * 80)
        print("SUCCESS: Figure 4(a) generation complete!")
        print("=" * 80)
        print()
        print("Output files:")
        print(f"  Data: {pickle_path}")
        print(f"  Figure: {figure_path}")
        print()
        print("Summary statistics:")
        print(f"  Train C (Zero-shot) - Median error at 120s: {plot_data['Train C']['median_all'][-1]:.2f} BPM")
        print(f"  Train C+B (Fine-tuning) - Median error at 120s: {plot_data['Train C+B']['median_all'][-1]:.2f} BPM")
        print("=" * 80)

        return 0

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return 1


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    exit(main())
