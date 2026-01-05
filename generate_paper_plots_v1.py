"""
Publication Figure Generator for LOSO Cross-Validation Results

Generates three publication-quality figures from Deep Corr-Encoder LOSO CV results:
- Figure (a): Waveform estimation (best/median/worst cases)
- Figure (b): MAE distribution box plot
- Figure (c): RR estimation scatter plot

Author: Zhantao Wang
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from existing codebase
from utils import CorrEncoder
from preprocess import CapnoBasePreprocessor
from loso_eval import (
    sliding_window_inference,
    calculate_respiratory_rate_fft,
    get_device
)

# ============================================================================
# Configuration
# ============================================================================

LOSO_RESULTS_DIR = 'loso_results'
RAW_DATA_DIR = 'raw_data'
WINDOW_SIZE = 288
STRIDE = 30
SAMPLING_RATE = 30  # Hz
BATCH_SIZE = 32

# Figure settings
FIGURE_DPI = 300


# ============================================================================
# Helper Functions: Data Loading
# ============================================================================

def load_all_fold_results():
    """Load the CSV file containing all fold metrics.

    Returns:
        pandas.DataFrame: DataFrame with columns: fold, subject_id, waveform_mae,
                         rr_error_bpm, duty_cycle_error, best_epoch, best_val_loss,
                         fold_time_minutes
    """
    csv_path = os.path.join(LOSO_RESULTS_DIR, 'all_fold_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def load_fold_model(fold_idx, device):
    """Load the best model checkpoint for a specific fold.

    Args:
        fold_idx: int (0-41)
        device: torch.device

    Returns:
        CorrEncoder model in eval mode, or None if loading fails
    """
    model_path = os.path.join(LOSO_RESULTS_DIR, f'fold_{fold_idx:02d}', 'best_model.pth')

    if not os.path.exists(model_path):
        warnings.warn(f"Model not found for fold {fold_idx}: {model_path}")
        return None

    try:
        model = CorrEncoder().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        warnings.warn(f"Failed to load model for fold {fold_idx}: {str(e)}")
        return None


def load_subject_continuous_signals(subject_id, preprocessor):
    """Load and preprocess continuous PPG and CO2 signals for a subject.

    Args:
        subject_id: str (e.g., "0009_8min")
        preprocessor: CapnoBasePreprocessor instance

    Returns:
        dict with keys 'ppg_continuous', 'co2_continuous' (both normalized numpy arrays)
    """
    # Load raw data
    ppg, co2, original_fs = preprocessor.load_subject_data(subject_id)

    # Apply preprocessing pipeline
    ppg_resampled = preprocessor.resample_signal(ppg, original_fs, preprocessor.target_fs)
    co2_resampled = preprocessor.resample_signal(co2, original_fs, preprocessor.target_fs)

    ppg_filtered = preprocessor.apply_highpass_filter(
        ppg_resampled, preprocessor.target_fs, cutoff=0.05, order=2
    )

    ppg_normalized = preprocessor.normalize_signal(ppg_filtered, -1, 1)
    co2_normalized = preprocessor.normalize_signal(co2_resampled, 0, 1)

    return {
        'ppg_continuous': ppg_normalized,
        'co2_continuous': co2_normalized
    }


def run_fold_inference(fold_idx, df, preprocessor, device):
    """Run inference for a specific fold and return predictions + ground truth.

    Args:
        fold_idx: int
        df: DataFrame from load_all_fold_results()
        preprocessor: CapnoBasePreprocessor instance
        device: torch.device

    Returns:
        dict with keys 'prediction', 'ground_truth', 'subject_id', or None if failed
    """
    try:
        # Get subject ID for this fold
        subject_id = df.loc[df['fold'] == fold_idx, 'subject_id'].values[0]

        # Load model
        model = load_fold_model(fold_idx, device)
        if model is None:
            return None

        # Load continuous signals
        signals = load_subject_continuous_signals(subject_id, preprocessor)

        # Run sliding window inference
        prediction = sliding_window_inference(
            model,
            signals['ppg_continuous'],
            device,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE
        )

        return {
            'prediction': prediction,
            'ground_truth': signals['co2_continuous'],
            'subject_id': subject_id
        }
    except Exception as e:
        warnings.warn(f"Inference failed for fold {fold_idx}: {str(e)}")
        return None


# ============================================================================
# Helper Functions: Figure Selection
# ============================================================================

def select_representative_folds(df):
    """Identify best, median, and worst folds based on waveform_mae.

    Args:
        df: DataFrame with 'fold' and 'waveform_mae' columns

    Returns:
        dict with keys 'best', 'median', 'worst' mapping to fold indices
    """
    # Sort by MAE
    sorted_df = df.sort_values('waveform_mae')

    best_fold = sorted_df.iloc[0]['fold']
    worst_fold = sorted_df.iloc[-1]['fold']

    # Median fold: middle of sorted list
    median_idx = len(sorted_df) // 2
    median_fold = sorted_df.iloc[median_idx]['fold']

    return {
        'best': int(best_fold),
        'median': int(median_fold),
        'worst': int(worst_fold)
    }


def select_display_segment(signal_length, segment_duration_sec=45, fs=30):
    """Select a visually interesting segment from the middle of the signal.

    Args:
        signal_length: int (total samples in continuous signal)
        segment_duration_sec: int (default 45 seconds for 30-60s range)
        fs: sampling rate

    Returns:
        tuple (start_idx, end_idx)
    """
    segment_samples = segment_duration_sec * fs  # e.g., 45 * 30 = 1350 samples

    # Center the segment
    center_idx = signal_length // 2
    start_idx = max(0, center_idx - segment_samples // 2)
    end_idx = min(signal_length, start_idx + segment_samples)

    # Adjust start if end hit boundary
    if end_idx - start_idx < segment_samples:
        start_idx = max(0, end_idx - segment_samples)

    return start_idx, end_idx


# ============================================================================
# Figure Generation Functions
# ============================================================================

def generate_figure_a_waveform(df, preprocessor, device, output_path):
    """Generate Figure (a): Waveform Estimation (Best, Median, Worst).

    Args:
        df: DataFrame from load_all_fold_results()
        preprocessor: CapnoBasePreprocessor instance
        device: torch.device
        output_path: str, path to save the figure
    """
    print("  Selecting representative folds...")
    rep_folds = select_representative_folds(df)
    print(f"  Selected folds: best={rep_folds['best']}, median={rep_folds['median']}, worst={rep_folds['worst']}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    fold_types = [
        ('best', 'Best Case'),
        ('median', 'Median Case'),
        ('worst', 'Worst Case')
    ]

    for idx, (fold_key, case_name) in enumerate(fold_types):
        fold_idx = rep_folds[fold_key]
        ax = axes[idx]

        print(f"  Processing {case_name} (fold {fold_idx})...")

        # Run inference
        result = run_fold_inference(fold_idx, df, preprocessor, device)

        if result is None:
            warnings.warn(f"Skipping {case_name} due to inference failure")
            ax.text(0.5, 0.5, f'{case_name}: Failed to load',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Extract segment
        signal_length = len(result['prediction'])
        start_idx, end_idx = select_display_segment(signal_length, segment_duration_sec=45)

        pred_segment = result['prediction'][start_idx:end_idx]
        gt_segment = result['ground_truth'][start_idx:end_idx]

        # Create time axis in seconds
        time_seconds = np.arange(len(pred_segment)) / SAMPLING_RATE

        # Get MAE from CSV
        mae_value = df.loc[df['fold'] == fold_idx, 'waveform_mae'].values[0]

        # Plot
        ax.plot(time_seconds, gt_segment, 'b-', linewidth=2, label='Ground Truth')
        ax.plot(time_seconds, pred_segment, color='orange', linestyle='--',
               linewidth=2, alpha=0.9, label='Prediction')

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Normalized CO2', fontsize=11)
        ax.set_title(f"Subject {result['subject_id']} ({case_name}): MAE = {mae_value:.4f}",
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Cleanup
        del result

    # Overall title
    fig.suptitle('Figure (a): CO2 Waveform Estimation Quality',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_figure_b_boxplot(df, output_path):
    """Generate Figure (b): MAE Box Plot.

    Args:
        df: DataFrame from load_all_fold_results()
        output_path: str, path to save the figure
    """
    # Extract MAE column
    mae_values = df['waveform_mae'].values
    median_mae = np.median(mae_values)

    print(f"  MAE statistics: median={median_mae:.4f}, mean={np.mean(mae_values):.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate box plot
    bp = ax.boxplot(
        [mae_values],
        positions=[1],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor='#87CEEB', edgecolor='darkblue', linewidth=1.5),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='darkblue', linewidth=1.5),
        capprops=dict(color='darkblue', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=8,
                       markeredgecolor='red', alpha=0.6)
    )

    # Add median annotation
    ax.text(1.35, median_mae, f'Median = {median_mae:.4f}',
           fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='gray', alpha=0.8))

    # Formatting
    ax.set_ylabel('Waveform MAE (Normalized CO2)', fontsize=12)
    ax.set_title('Figure (b): MAE Distribution Across 42 LOSO Folds',
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_figure_c_scatter(df, preprocessor, device, output_path):
    """Generate Figure (c): RR Scatter Plot.

    Args:
        df: DataFrame from load_all_fold_results()
        preprocessor: CapnoBasePreprocessor instance
        device: torch.device
        output_path: str, path to save the figure
    """
    rr_true_list = []
    rr_pred_list = []
    colors = []
    subject_ids = []

    print("  Calculating RR for all 42 folds...")
    for fold_idx in tqdm(range(42), desc="  Processing folds"):
        result = run_fold_inference(fold_idx, df, preprocessor, device)

        if result is None:
            warnings.warn(f"Skipping fold {fold_idx} due to inference failure")
            continue

        # Calculate RR
        rr_true = calculate_respiratory_rate_fft(result['ground_truth'], fs=SAMPLING_RATE)
        rr_pred = calculate_respiratory_rate_fft(result['prediction'], fs=SAMPLING_RATE)

        # Skip if RR is 0 (edge case)
        if rr_true == 0 or rr_pred == 0:
            warnings.warn(f"Zero RR detected for fold {fold_idx}, skipping")
            del result
            continue

        # Determine color
        error = abs(rr_pred - rr_true)
        color = 'red' if error > 20 else 'blue'

        rr_true_list.append(rr_true)
        rr_pred_list.append(rr_pred)
        colors.append(color)
        subject_ids.append(result['subject_id'])

        # Memory cleanup
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Check if we have enough data
    if len(rr_true_list) < 30:
        raise RuntimeError(f"Only {len(rr_true_list)}/42 folds succeeded. Need at least 30.")

    print(f"  Successfully processed {len(rr_true_list)}/42 folds")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Separate points by color using boolean masking
    colors_array = np.array(colors)
    blue_mask = colors_array == 'blue'
    red_mask = colors_array == 'red'

    rr_true_array = np.array(rr_true_list)
    rr_pred_array = np.array(rr_pred_list)

    # Plot blue points (good estimates)
    if np.any(blue_mask):
        ax.scatter(
            rr_true_array[blue_mask],
            rr_pred_array[blue_mask],
            c='blue', s=60, alpha=0.6, edgecolors='black', linewidths=0.5,
            label=f'Good (|error| ≤ 20 BPM, N={np.sum(blue_mask)})'
        )

    # Plot red points (outliers) on top
    if np.any(red_mask):
        ax.scatter(
            rr_true_array[red_mask],
            rr_pred_array[red_mask],
            c='red', s=60, alpha=0.6, edgecolors='black', linewidths=0.5,
            label=f'Outlier (|error| > 20 BPM, N={np.sum(red_mask)})'
        )

    # Diagonal line y=x
    min_val = min(rr_true_list)
    max_val = max(max(rr_true_list), max(rr_pred_list))
    ax.plot([min_val, max_val], [min_val, max_val],
           'k--', linewidth=2, alpha=0.5, label='Perfect Estimation')

    # Calculate and display statistics
    mae = np.mean(np.abs(rr_pred_array - rr_true_array))
    corr = np.corrcoef(rr_true_list, rr_pred_list)[0, 1]

    stats_text = f'MAE: {mae:.2f} BPM\nCorrelation: {corr:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatting
    ax.set_xlabel('Reference RR (BPM)', fontsize=12)
    ax.set_ylabel('Estimated RR (BPM)', fontsize=12)
    ax.set_title('Figure (c): Respiratory Rate Estimation Accuracy',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("Publication Figure Generator for LOSO Cross-Validation Results")
    print("=" * 80)
    print()

    # Configure plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
    })

    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    print()

    # Load results CSV
    print("Loading LOSO results...")
    df = load_all_fold_results()
    print(f"Loaded {len(df)} fold results")
    print()

    # Initialize preprocessor
    preprocessor = CapnoBasePreprocessor(RAW_DATA_DIR, 'processed_data')

    # Generate Figure (a): Waveform Estimation
    print("Generating Figure (a): Waveform Estimation...")
    output_a = os.path.join(LOSO_RESULTS_DIR, 'figa_waveform_estimation.png')
    generate_figure_a_waveform(df, preprocessor, device, output_a)
    print()

    # Generate Figure (b): MAE Box Plot
    print("Generating Figure (b): MAE Box Plot...")
    output_b = os.path.join(LOSO_RESULTS_DIR, 'figb_mae_boxplot.png')
    generate_figure_b_boxplot(df, output_b)
    print()

    # Generate Figure (c): RR Scatter Plot
    print("Generating Figure (c): RR Scatter Plot...")
    output_c = os.path.join(LOSO_RESULTS_DIR, 'figc_rr_scatter.png')
    generate_figure_c_scatter(df, preprocessor, device, output_c)
    print()

    print("=" * 80)
    print("All figures generated successfully!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {output_a}")
    print(f"  - {output_b}")
    print(f"  - {output_c}")


if __name__ == "__main__":
    main()
