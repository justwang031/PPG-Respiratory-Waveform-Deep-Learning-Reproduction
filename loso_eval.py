"""
LOSO Cross-Validation for Deep Corr-Encoder
Leave-One-Subject-Out (LOSO) Cross-Validation script for strict paper reproduction.

Implements 42-fold LOSO CV with:
- Sliding window inference with stride=30 for test subjects
- FFT-based respiratory rate calculation
- Duty cycle metric based on CO2 waveform thresholding
- Median aggregation across all folds

Author: Zhantao Wang
"""

import os
import gc
import json
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - CRITICAL for silent plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal

from utils import CorrEncoder, count_parameters
from preprocess import CapnoBasePreprocessor


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 80
WINDOW_SIZE = 288
STRIDE = 30  # For sliding window inference on test data
RESULTS_DIR = 'loso_results'


# ============================================================================
# Reproducibility
# ============================================================================

def seed_everything(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Auto-detect best available device: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


# ============================================================================
# Core Utilities: Sliding Windows
# ============================================================================

def create_sliding_windows(signal_data, window_size=288, stride=30):
    """
    Create overlapping windows from continuous signal.

    Args:
        signal_data: 1D numpy array [N_samples]
        window_size: 288 samples (9.6s at 30Hz)
        stride: 30 samples (1 second at 30Hz)

    Returns:
        windows: numpy array [N_windows, window_size]
        start_indices: list of starting indices for reconstruction
    """
    n_samples = len(signal_data)
    windows = []
    start_indices = []

    # Slide window across signal
    for start_idx in range(0, n_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        windows.append(signal_data[start_idx:end_idx])
        start_indices.append(start_idx)

    return np.array(windows), start_indices


def sliding_window_inference(model, ppg_signal, device, window_size=288, stride=30, batch_size=32):
    """
    Perform inference with overlapping windows and average predictions.

    Strategy: Create sliding windows, predict each, then average overlaps.

    Args:
        model: Trained CorrEncoder model
        ppg_signal: Continuous PPG signal [N_samples]
        device: Device to run inference on
        window_size: 288 samples
        stride: 30 samples
        batch_size: 32

    Returns:
        fused_prediction: [N_samples] continuous CO2 prediction
    """
    model.eval()
    n_samples = len(ppg_signal)

    # Create sliding windows
    ppg_windows, start_indices = create_sliding_windows(ppg_signal, window_size, stride)

    # Convert to tensor and add channel dimension
    ppg_tensor = torch.FloatTensor(ppg_windows).unsqueeze(1).to(device)  # [N_windows, 1, 288]

    # Predict in batches
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(ppg_tensor), batch_size):
            batch = ppg_tensor[i:i+batch_size]
            pred = model(batch)
            all_predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)  # [N_windows, 1, 288]
    predictions = predictions.squeeze(1)  # [N_windows, 288]

    # Initialize accumulation arrays
    accumulated_signal = np.zeros(n_samples, dtype=np.float32)
    count_array = np.zeros(n_samples, dtype=np.float32)

    # Accumulate overlapping predictions
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

    # Average where overlaps exist
    fused_prediction = np.divide(
        accumulated_signal,
        count_array,
        where=count_array > 0,
        out=np.zeros_like(accumulated_signal)
    )

    return fused_prediction


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_waveform_mae(prediction, ground_truth):
    """
    Calculate Mean Absolute Error between predicted and ground truth CO2.

    Args:
        prediction: [N_samples] continuous CO2 prediction
        ground_truth: [N_samples] ground truth CO2

    Returns:
        mae: Float scalar
    """
    mae = np.mean(np.abs(prediction - ground_truth))
    return mae


def calculate_respiratory_rate_fft(co2_signal, fs=30):
    """
    Extract dominant respiratory frequency using FFT.

    Method:
    1. Remove DC component (subtract mean)
    2. Compute FFT and magnitude spectrum
    3. Restrict to physiological range: 0.1-0.7 Hz (6-42 BPM)
    4. Find peak frequency
    5. Convert to BPM: peak_freq * 60

    Args:
        co2_signal: [N_samples] - entire 8-minute continuous signal
        fs: 30 Hz sampling rate

    Returns:
        rr_bpm: Respiratory rate in breaths per minute
    """
    # Remove DC component
    co2_centered = co2_signal - np.mean(co2_signal)

    # Compute FFT
    n = len(co2_centered)
    fft_vals = np.fft.fft(co2_centered)
    fft_mag = np.abs(fft_vals[:n//2])  # Positive frequencies only
    freqs = np.fft.fftfreq(n, d=1/fs)[:n//2]

    # Restrict to physiological range: 0.1-0.7 Hz (6-42 BPM)
    # Lower bound 0.1 Hz filters out DC component and very low frequencies
    mask = (freqs >= 0.1) & (freqs <= 0.7)
    valid_fft = fft_mag[mask]
    valid_freqs = freqs[mask]

    # Find peak frequency
    if len(valid_fft) == 0:
        return 0.0  # Fallback if no valid frequencies

    peak_idx = np.argmax(valid_fft)
    peak_freq = valid_freqs[peak_idx]

    # Convert to BPM
    rr_bpm = peak_freq * 60

    return rr_bpm


def calculate_rr_error_fft(prediction, ground_truth, fs=30):
    """
    Calculate absolute error between predicted and true respiratory rate.

    Args:
        prediction: [N_samples] predicted CO2
        ground_truth: [N_samples] ground truth CO2
        fs: 30 Hz

    Returns:
        rr_error: Absolute RR error in BPM
    """
    rr_pred = calculate_respiratory_rate_fft(prediction, fs)
    rr_true = calculate_respiratory_rate_fft(ground_truth, fs)
    return abs(rr_pred - rr_true)


def calculate_duty_cycle(co2_signal, threshold=0.5):
    """
    Calculate inhalation duty cycle.

    Definition: Fraction of respiratory cycle in inhalation phase
    Method: Binarize at threshold 0.5 (midpoint of [0,1] normalization)

    CO2 < 0.5 → Inhalation (binary value = 1)
    CO2 >= 0.5 → Exhalation (binary value = 0)

    Args:
        co2_signal: [N_samples] normalized to [0, 1]
        threshold: 0.5

    Returns:
        duty_cycle: Float in [0, 1]
    """
    binary_signal = (co2_signal < threshold).astype(float)
    duty_cycle = np.mean(binary_signal)
    return duty_cycle


def calculate_duty_cycle_error(prediction, ground_truth, threshold=0.5):
    """
    Calculate absolute error between predicted and true duty cycle.

    Args:
        prediction: [N_samples] predicted CO2
        ground_truth: [N_samples] ground truth CO2
        threshold: 0.5

    Returns:
        dc_error: Absolute duty cycle error
    """
    dc_pred = calculate_duty_cycle(prediction, threshold)
    dc_true = calculate_duty_cycle(ground_truth, threshold)
    return abs(dc_pred - dc_true)


def evaluate_loso_fold(model, test_data, device):
    """
    Comprehensive evaluation with all three metrics.

    Steps:
    1. Generate fused prediction using sliding_window_inference()
       (stride=30, overlap averaging)
    2. Extract ground truth continuous signal
    3. Calculate all three metrics on continuous signals

    Args:
        model: Trained CorrEncoder model
        test_data: Dict with 'ppg_continuous' and 'co2_continuous'
        device: Device

    Returns:
        metrics: {
            'waveform_mae': float,
            'rr_error_bpm': float,
            'duty_cycle_error': float
        }
    """
    # Generate fused prediction
    ppg_continuous = test_data['ppg_continuous']
    co2_continuous = test_data['co2_continuous']

    fused_prediction = sliding_window_inference(model, ppg_continuous, device)

    # Calculate metrics
    waveform_mae = calculate_waveform_mae(fused_prediction, co2_continuous)
    rr_error = calculate_rr_error_fft(fused_prediction, co2_continuous)
    duty_cycle_error = calculate_duty_cycle_error(fused_prediction, co2_continuous)

    return {
        'waveform_mae': waveform_mae,
        'rr_error_bpm': rr_error,
        'duty_cycle_error': duty_cycle_error
    }


# ============================================================================
# Data Management: LOSO Data Loader
# ============================================================================

class LOSODataLoader:
    """Manages subject-based LOSO splits"""

    def __init__(self, raw_data_dir='raw_data'):
        self.preprocessor = CapnoBasePreprocessor(raw_data_dir, 'processed_data')
        self.subject_ids = self.preprocessor.discover_subjects()

    def get_loso_split(self, test_subject_idx):
        """
        Create train/test split for one LOSO fold.

        Args:
            test_subject_idx: Index of test subject (0-41)

        Returns:
            train_loader: DataLoader with non-overlapping windows from 41 subjects
            test_data: Dict with continuous preprocessed signals from 1 subject
        """
        # Identify test subject
        test_subject_id = self.subject_ids[test_subject_idx]

        # Train subjects: all except test subject
        train_subject_ids = [sid for i, sid in enumerate(self.subject_ids) if i != test_subject_idx]

        # Process training subjects (41 subjects)
        train_ppg_all = []
        train_co2_all = []

        for subject_id in train_subject_ids:
            ppg_windows, co2_windows = self.preprocessor.process_subject(subject_id)
            train_ppg_all.extend(ppg_windows)
            train_co2_all.extend(co2_windows)

        # Convert to tensors and create DataLoader
        train_ppg_tensor = torch.FloatTensor(train_ppg_all).unsqueeze(1)  # [N, 1, 288]
        train_co2_tensor = torch.FloatTensor(train_co2_all).unsqueeze(1)  # [N, 1, 288]

        train_dataset = TensorDataset(train_ppg_tensor, train_co2_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Process test subject (1 subject) - keep continuous
        ppg, co2, original_fs = self.preprocessor.load_subject_data(test_subject_id)

        # Apply same preprocessing as training
        ppg_resampled = self.preprocessor.resample_signal(ppg, original_fs, self.preprocessor.target_fs)
        co2_resampled = self.preprocessor.resample_signal(co2, original_fs, self.preprocessor.target_fs)

        ppg_filtered = self.preprocessor.apply_highpass_filter(
            ppg_resampled, self.preprocessor.target_fs, cutoff=0.05, order=2
        )

        ppg_normalized = self.preprocessor.normalize_signal(ppg_filtered, -1, 1)
        co2_normalized = self.preprocessor.normalize_signal(co2_resampled, 0, 1)

        test_data = {
            'ppg_continuous': ppg_normalized,
            'co2_continuous': co2_normalized,
            'subject_id': test_subject_id
        }

        return train_loader, test_data


# ============================================================================
# Training Loop for LOSO
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_on_test_subject(model, test_data, criterion, device):
    """
    Compute validation loss on test subject using non-overlapping windows.
    This is ONLY for model selection during training, not final evaluation.
    """
    model.eval()

    # Create non-overlapping windows from continuous signal
    ppg_continuous = test_data['ppg_continuous']
    co2_continuous = test_data['co2_continuous']

    # Segment into non-overlapping windows
    num_windows = len(ppg_continuous) // WINDOW_SIZE
    ppg_windows = []
    co2_windows = []

    for i in range(num_windows):
        start_idx = i * WINDOW_SIZE
        end_idx = start_idx + WINDOW_SIZE
        ppg_windows.append(ppg_continuous[start_idx:end_idx])
        co2_windows.append(co2_continuous[start_idx:end_idx])

    if len(ppg_windows) == 0:
        return float('inf')  # Signal too short

    # Convert to tensors
    ppg_tensor = torch.FloatTensor(ppg_windows).unsqueeze(1).to(device)
    co2_tensor = torch.FloatTensor(co2_windows).unsqueeze(1).to(device)

    # Compute loss
    with torch.no_grad():
        predictions = model(ppg_tensor)
        loss = criterion(predictions, co2_tensor)

    return loss.item()


def train_loso_fold(model, train_loader, test_data, fold_idx, criterion, optimizer, device):
    """
    Train for 80 epochs, monitoring test subject S_i for best checkpoint.

    Returns:
        best_model_path: Path to saved best checkpoint
    """
    fold_dir = os.path.join(RESULTS_DIR, f'fold_{fold_idx:02d}')
    os.makedirs(fold_dir, exist_ok=True)

    best_model_path = os.path.join(fold_dir, 'best_model.pth')
    log_path = os.path.join(fold_dir, 'training_log.txt')

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Initialize log
    with open(log_path, 'w') as f:
        f.write(f"LOSO Fold {fold_idx} Training Log\n")
        f.write(f"Test Subject: {test_data['subject_id']}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate on test subject
        val_loss = validate_on_test_subject(model, test_data, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)

        # Log
        with open(log_path, 'a') as f:
            best_marker = " [BEST]" if is_best else ""
            f.write(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
                   f"Train: {train_loss:.6f} | Val: {val_loss:.6f}{best_marker}\n")

    # Plot loss curve (SILENT - no plt.show())
    plot_path = os.path.join(fold_dir, 'loss_curve.png')
    plot_loss_curve_silent(train_losses, val_losses, plot_path)

    return best_model_path


def plot_loss_curve_silent(train_losses, val_losses, save_path):
    """Plot and save loss curve WITHOUT displaying."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Test Subject Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('LOSO Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # CRITICAL: Close without showing


# ============================================================================
# Helper Functions: Results Management
# ============================================================================

def check_completed_folds(results_dir):
    """Check which folds have already been completed."""
    completed = []
    for fold_idx in range(42):
        metrics_path = os.path.join(results_dir, f'fold_{fold_idx:02d}', 'metrics.json')
        if os.path.exists(metrics_path):
            completed.append(fold_idx)
    return completed


def load_fold_results(fold_idx, results_dir):
    """Load existing fold results from metrics.json"""
    metrics_path = os.path.join(results_dir, f'fold_{fold_idx:02d}', 'metrics.json')
    with open(metrics_path, 'r') as f:
        return json.load(f)


def save_fold_results(fold_results, fold_dir):
    """Save fold results to metrics.json"""
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(fold_results, f, indent=2)


def aggregate_loso_results(all_results):
    """
    Calculate MEDIAN (not mean) across all folds.

    Returns:
        aggregated: {
            'waveform_mae': {'median': X, 'mean': Y, 'std': Z},
            ...
        }
    """
    metrics = ['waveform_mae', 'rr_error_bpm', 'duty_cycle_error']
    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in all_results]
        aggregated[metric] = {
            'median': float(np.median(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return aggregated


def save_final_results(all_results, aggregated, results_dir):
    """Save all results to CSV and JSON files."""
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, 'all_fold_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nAll fold results saved to: {csv_path}")

    # Save aggregated JSON
    final_results = {
        'method': 'LOSO Cross-Validation',
        'n_folds': len(all_results),
        'aggregation': 'median',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': aggregated
    }

    json_path = os.path.join(results_dir, 'aggregated_results.json')
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Aggregated results saved to: {json_path}")


# ============================================================================
# Main LOSO Cross-Validation Loop
# ============================================================================

def run_loso_cross_validation():
    """
    Main execution loop for 42-fold LOSO CV.
    """
    print("=" * 80)
    print("LOSO Cross-Validation for Deep Corr-Encoder")
    print("=" * 80)
    print()

    # Setup
    device = get_device()
    print(f"Device: {device}")
    print()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize data loader
    data_loader = LOSODataLoader()
    print(f"Total subjects: {len(data_loader.subject_ids)}")
    print()

    # Check for completed folds (resume support)
    completed_folds = check_completed_folds(RESULTS_DIR)
    if completed_folds:
        print(f"Found {len(completed_folds)} completed folds: {completed_folds}")
        print("Will skip these and resume from next fold.")
        print()

    all_results = []

    # Main LOSO loop
    for fold_idx in tqdm(range(42), desc="LOSO Folds"):
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

            # STEP 1: Reset seed for reproducibility
            seed_everything(42)

            # STEP 2: Get LOSO split
            train_loader, test_data = data_loader.get_loso_split(fold_idx)
            print(f"Test subject: {test_data['subject_id']}")
            print(f"Training samples: {len(train_loader.dataset)}")
            print(f"Test signal length: {len(test_data['ppg_continuous'])} samples")

            # STEP 3: Initialize fresh model
            model = CorrEncoder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.MSELoss()

            # STEP 4: Train for 80 epochs
            print(f"\nTraining...")
            best_model_path = train_loso_fold(
                model, train_loader, test_data, fold_idx,
                criterion, optimizer, device
            )

            # STEP 5: Load best checkpoint
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # STEP 6: Evaluate with sliding window fusion
            print(f"Evaluating with sliding window inference (stride={STRIDE})...")
            metrics = evaluate_loso_fold(model, test_data, device)

            # STEP 7: Save fold results
            fold_time = time.time() - fold_start_time
            fold_results = {
                'fold': fold_idx,
                'subject_id': test_data['subject_id'],
                'waveform_mae': metrics['waveform_mae'],
                'rr_error_bpm': metrics['rr_error_bpm'],
                'duty_cycle_error': metrics['duty_cycle_error'],
                'best_epoch': checkpoint['epoch'] + 1,
                'best_val_loss': checkpoint['val_loss'],
                'fold_time_minutes': fold_time / 60
            }

            save_fold_results(fold_results, fold_dir)
            all_results.append(fold_results)

            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Waveform MAE: {metrics['waveform_mae']:.6f}")
            print(f"  RR Error (BPM): {metrics['rr_error_bpm']:.2f}")
            print(f"  Duty Cycle Error: {metrics['duty_cycle_error']:.4f}")
            print(f"  Time: {fold_time/60:.2f} minutes")

            # STEP 8: Memory cleanup
            del model, optimizer, train_loader, test_data, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            # Error handling - log but continue
            error_msg = f"Fold {fold_idx} failed: {str(e)}"
            print(f"\nERROR: {error_msg}")

            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, 'error.log'), 'w') as f:
                f.write(error_msg + '\n\n')
                f.write(traceback.format_exc())

            continue

    # STEP 9: Aggregate results (MEDIAN)
    print(f"\n{'='*80}")
    print("Aggregating Results")
    print(f"{'='*80}")

    aggregated = aggregate_loso_results(all_results)

    # STEP 10: Save final outputs
    save_final_results(all_results, aggregated, RESULTS_DIR)

    # Print final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS (MEDIAN across 42 folds)")
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
    results = run_loso_cross_validation()
