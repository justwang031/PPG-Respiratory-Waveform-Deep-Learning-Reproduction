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
- Uses pre-windowed data (no sliding window inference needed)
- Simplified evaluation: concatenate windows → calculate RR via FFT

Author: Deep Learning Engineer
Date: 2025-12-12
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core_model.utils import CorrEncoder, count_parameters
from evaluation.loso_eval import calculate_respiratory_rate_fft, seed_everything, get_device


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

# Inference batch size (for efficient GPU usage)
BATCH_SIZE = 32

# Reproducibility
RANDOM_SEED = 42


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

def evaluate_subject(model, subject_idx, ppg_windows, resp_windows, device):
    """
    Evaluate model on a single BIDMC subject.

    Strategy (simplified from sliding window):
    1. Pass 50 pre-windowed PPG samples through model
    2. Flatten predictions: [50, 288] → [14400] pseudo-continuous signal
    3. Flatten ground truth: [50, 288] → [14400] pseudo-continuous signal
    4. Calculate respiratory rate using FFT on both signals
    5. Compute absolute RR error

    Args:
        model: Pre-trained CorrEncoder in eval mode
        subject_idx: Subject index (0-52)
        ppg_windows: [50, 288] PPG windows for this subject
        resp_windows: [50, 288] RESP windows for this subject
        device: Device to run inference on

    Returns:
        Dict with evaluation metrics
    """
    # Add channel dimension: [50, 288] → [50, 1, 288]
    ppg_tensor = ppg_windows.unsqueeze(1).to(device)

    # Run inference in batches (more efficient for GPU)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(ppg_tensor), BATCH_SIZE):
            batch = ppg_tensor[i:i+BATCH_SIZE]
            pred = model(batch)  # [batch_size, 1, 288]
            predictions.append(pred.cpu())

    # Concatenate batches: [50, 1, 288]
    predictions = torch.cat(predictions, dim=0)

    # Remove channel dimension: [50, 1, 288] → [50, 288]
    predictions = predictions.squeeze(1)

    # Flatten to pseudo-continuous signals: [50, 288] → [14400]
    pred_continuous = predictions.reshape(-1).numpy()  # [14400]
    resp_continuous = resp_windows.reshape(-1).numpy()  # [14400]

    # Calculate respiratory rate using FFT
    rr_pred = calculate_respiratory_rate_fft(pred_continuous, fs=SAMPLING_RATE)
    rr_true = calculate_respiratory_rate_fft(resp_continuous, fs=SAMPLING_RATE)

    # Calculate absolute error
    rr_error = abs(rr_pred - rr_true)

    # Format subject_id as zero-padded string: "01", "02", ..., "53"
    subject_id_str = f"{subject_idx + 1:02d}"

    return {
        'subject_id': subject_id_str,
        'rr_true': rr_true,
        'rr_pred': rr_pred,
        'rr_error': rr_error
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
    print("Respiratory Rate Error Statistics:")
    print(f"  Mean (MAE):  {df['rr_error'].mean():.2f} BPM")
    print(f"  Std Dev:     {df['rr_error'].std():.2f} BPM")
    print(f"  Median:      {df['rr_error'].median():.2f} BPM")
    print(f"  Min:         {df['rr_error'].min():.2f} BPM")
    print(f"  Max:         {df['rr_error'].max():.2f} BPM")
    print("=" * 80)
    print()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    os.chdir(ROOT)
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
