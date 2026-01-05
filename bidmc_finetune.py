"""
Fine-Tuning Evaluation of CapnoBase-Trained Model on BIDMC Dataset

Performs Leave-One-Subject-Out (LOSO) cross-validation with transfer learning:
- Loads pre-trained CapnoBase model weights (capnobase_master.pth)
- For each fold: Fine-tunes on 52 subjects, evaluates on 1 held-out subject
- Uses lower learning rate (1e-4) and fewer epochs (15) to preserve pre-trained features

This script demonstrates Scenario B: Transfer Learning
- Source domain: CapnoBase (PPG → CO2)
- Target domain: BIDMC (PPG → RESP)
- Strategy: Fine-tuning (Transfer Learning)

Key differences from zero-shot:
- Zero-shot: Load model → Evaluate (no training)
- Fine-tuning: Load model → Train on 52 subjects → Evaluate on 1 subject

Evaluation metrics: Respiratory rate MAE on 30.6s and 60.6s windows

Author: Zhantao Wang
"""

import os
import gc
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from utils import CorrEncoder, count_parameters
from loso_eval import calculate_respiratory_rate_fft, seed_everything, get_device, sliding_window_inference


# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint path (pre-trained on CapnoBase)
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

# Fine-tuning hyperparameters
NUM_EPOCHS = 15  # Fewer epochs than training from scratch (80)
LEARNING_RATE = 1e-4  # Lower LR than training from scratch (1e-3)
BATCH_SIZE = 32

# Reproducibility
RANDOM_SEED = 42

# Evaluation window sizes (matching paper specification)
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
# Evaluation Helper Functions
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


# ============================================================================
# Training Function
# ============================================================================

def train_fold(model, train_loader, optimizer, criterion, device, num_epochs=NUM_EPOCHS):
    """
    Fine-tune model for specified number of epochs.

    Args:
        model: CorrEncoder model initialized with pre-trained weights
        train_loader: DataLoader with training data (52 subjects)
        optimizer: Adam optimizer with LR=1e-4
        criterion: MSELoss
        device: Device to train on
        num_epochs: Number of epochs (default: 15)

    Returns:
        final_loss: Training loss from final epoch
    """
    model.train()
    final_loss = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

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

            epoch_loss += loss.item()
            num_batches += 1

        final_loss = epoch_loss / num_batches

    return final_loss


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_subject(model, test_ppg, test_resp, subject_idx, device):
    """
    Evaluate fine-tuned model on a single BIDMC subject.

    Strategy (using sliding window inference):
    1. Flatten 50 pre-windowed PPG samples into continuous signal [14400]
    2. Apply sliding window inference with overlap-add (stride=30)
    3. Flatten 50 ground truth RESP windows into continuous signal [14400]
    4. Calculate respiratory rate MAE for 30.6s windows (918 samples)
    5. Calculate respiratory rate MAE for 60.6s windows (1818 samples)

    Args:
        model: Fine-tuned CorrEncoder in eval mode
        test_ppg: [50, 288] PPG windows for this subject
        test_resp: [50, 288] RESP windows for this subject
        subject_idx: Subject index (0-52)
        device: Device to run inference on

    Returns:
        Dict with evaluation metrics (subject_id, mae_30s, mae_60s)
    """
    # ========================================================================
    # Step 1: Reconstruct Continuous Signals from Pre-windowed Data
    # ========================================================================
    ppg_continuous = test_ppg.reshape(-1).numpy()  # [14400]
    resp_continuous = test_resp.reshape(-1).numpy()  # [14400]

    # ========================================================================
    # Step 2: Sliding Window Inference with Overlap-Add
    # ========================================================================
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
# Main LOSO Cross-Validation Loop
# ============================================================================

def run_finetune_loso():
    """
    Main execution function for LOSO fine-tuning on BIDMC dataset.

    Steps:
    1. Load BIDMC data (2650 windows)
    2. Loop through 53 subjects (LOSO folds):
       - Split: 52 subjects for training, 1 for testing
       - Initialize model and load pre-trained CapnoBase weights
       - Fine-tune for 15 epochs with LR=1e-4
       - Evaluate on held-out subject
       - Calculate RR error via FFT on 30.6s and 60.6s windows
    3. Save results to CSV
    4. Print summary statistics
    """
    # Set random seeds for reproducibility
    seed_everything(RANDOM_SEED)

    print("=" * 80)
    print("BIDMC TRANSFER LEARNING (SCENARIO B: FINE-TUNING)")
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

    # Storage for results
    all_results = []

    # LOSO loop
    print("=" * 80)
    print("LOSO Cross-Validation: Fine-Tuning on 52 Subjects, Testing on 1")
    print("=" * 80)
    print()

    start_time = time.time()

    for subject_idx in tqdm(range(NUM_SUBJECTS), desc="LOSO Folds"):
        # ====================================================================
        # Step 1: Data Splitting (LOSO)
        # ====================================================================
        test_start = subject_idx * WINDOWS_PER_SUBJECT
        test_end = (subject_idx + 1) * WINDOWS_PER_SUBJECT

        test_ppg = samples[test_start:test_end]      # [50, 288]
        test_resp = labels[test_start:test_end]      # [50, 288]

        # Concatenate all other subjects for training
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
            train_ppg = torch.cat([
                samples[:test_start],
                samples[test_end:]
            ], dim=0)
            train_resp = torch.cat([
                labels[:test_start],
                labels[test_end:]
            ], dim=0)

        # Verify training set size (should be 2600 = 52 * 50)
        expected_train_size = (NUM_SUBJECTS - 1) * WINDOWS_PER_SUBJECT
        assert train_ppg.shape[0] == expected_train_size, \
            f"Expected {expected_train_size} training samples, got {train_ppg.shape[0]}"

        # ====================================================================
        # Step 2: Create DataLoader for Training
        # ====================================================================
        # Add channel dimension for CorrEncoder: [N, 288] -> [N, 1, 288]
        train_dataset = TensorDataset(
            train_ppg.unsqueeze(1),
            train_resp.unsqueeze(1)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # ====================================================================
        # Step 3: Model Initialization with Transfer Learning
        # ====================================================================
        # Initialize fresh model
        model = CorrEncoder().to(device)

        # CRITICAL: Load pre-trained weights from CapnoBase
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # ====================================================================
        # Step 4: Fine-Tuning Phase
        # ====================================================================
        final_loss = train_fold(model, train_loader, optimizer, criterion, device)

        # ====================================================================
        # Step 5: Evaluation Phase
        # ====================================================================
        model.eval()
        result = evaluate_subject(model, test_ppg, test_resp, subject_idx, device)
        all_results.append(result)

        # ====================================================================
        # Step 6: Memory Cleanup
        # ====================================================================
        del model, optimizer, train_loader, train_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
    csv_path = os.path.join(RESULTS_DIR, 'finetune_results.csv')
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
    print("FINE-TUNING SUMMARY")
    print("=" * 80)
    print(f"Transfer Learning:   CapnoBase → BIDMC")
    print(f"Model Checkpoint:    {MODEL_CHECKPOINT}")
    print(f"Fine-tuning Config:  LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}")
    print(f"Subjects Evaluated:  {len(all_results)}")
    print(f"Total Time:          {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
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
    print(f"  Median MAE (mMAE):  {df['mae_60s'].median():.2f} BPM")
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
        results = run_finetune_loso()
        print("Fine-tuning evaluation completed successfully!")

    except FileNotFoundError as e:
        print(f"ERROR: Required file not found - {e}")
        print(f"Please ensure:")
        print(f"  1. {MODEL_CHECKPOINT} exists (run train_master_model.py)")
        print(f"  2. {BIDMC_DATA_PATH} exists (run preprocess_bidmc.py)")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(traceback.format_exc())
