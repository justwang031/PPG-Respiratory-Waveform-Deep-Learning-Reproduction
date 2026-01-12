"""
Exploratory Data Analysis (EDA) Script for Deep Corr-Encoder Dataset

This script performs comprehensive EDA on the processed CapnoBase dataset,
including waveform visualization and respiratory rate distribution analysis.

Author: Zhantao Wang
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set seaborn style for nicer plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class DatasetEDA:
    """Exploratory Data Analysis for processed PyTorch dataset"""

    def __init__(self, processed_data_dir='processed_data', output_dir='eda_results'):
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir
        self.waveforms_dir = os.path.join(output_dir, 'waveforms')

        # Dataset parameters (from preprocessing script)
        self.target_fs = 30  # Sampling rate in Hz
        self.window_duration = 9.6  # Window duration in seconds
        self.window_samples = int(self.target_fs * self.window_duration)  # 288 samples

        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.waveforms_dir).mkdir(parents=True, exist_ok=True)

    def load_datasets(self):
        """
        Load the processed train and test datasets.

        Returns:
            train_data: Dictionary with 'samples' and 'labels'
            test_data: Dictionary with 'samples' and 'labels'
        """
        print("=" * 60)
        print("Loading Processed Datasets")
        print("=" * 60)

        train_path = os.path.join(self.processed_data_dir, 'train.pt')
        test_path = os.path.join(self.processed_data_dir, 'test.pt')

        train_data = torch.load(train_path)
        test_data = torch.load(test_path)

        # Print shapes for verification
        print(f"\nTrain Dataset:")
        print(f"  Samples (PPG): {train_data['samples'].shape}")
        print(f"  Labels (CO2):  {train_data['labels'].shape}")

        print(f"\nTest Dataset:")
        print(f"  Samples (PPG): {test_data['samples'].shape}")
        print(f"  Labels (CO2):  {test_data['labels'].shape}")

        print(f"\nWindow Parameters:")
        print(f"  Duration: {self.window_duration}s")
        print(f"  Sampling Rate: {self.target_fs}Hz")
        print(f"  Samples per Window: {self.window_samples}")

        return train_data, test_data

    def plot_waveform(self, ppg, co2, save_path, title_prefix="Sample"):
        """
        Plot a single PPG-CO2 waveform pair.

        Args:
            ppg: PPG signal array (288 samples)
            co2: CO2 signal array (288 samples)
            save_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        # Create time axis
        time = np.arange(len(ppg)) / self.target_fs

        # Create figure with 2 subplots sharing x-axis
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot PPG (Input)
        axes[0].plot(time, ppg, color='blue', linewidth=1.5)
        axes[0].set_ylabel('PPG Signal', fontsize=11, fontweight='bold')
        axes[0].set_title(f'{title_prefix} - PPG Input [Min: {ppg.min():.3f}, Max: {ppg.max():.3f}]',
                         fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot CO2 (Target)
        axes[1].plot(time, co2, color='orange', linewidth=1.5)
        axes[1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('CO2 Signal', fontsize=11, fontweight='bold')
        axes[1].set_title(f'{title_prefix} - CO2 Target [Min: {co2.min():.3f}, Max: {co2.max():.3f}]',
                         fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Free memory

    def visualize_waveform_gallery(self, train_data, test_data, num_samples=50):
        """
        Generate waveform gallery by randomly sampling windows.

        Args:
            train_data: Training dataset dictionary
            test_data: Testing dataset dictionary
            num_samples: Number of samples to plot per dataset
        """
        print("\n" + "=" * 60)
        print("Generating Waveform Gallery")
        print("=" * 60)

        # Get data as numpy arrays
        train_ppg = train_data['samples'].numpy()
        train_co2 = train_data['labels'].numpy()
        test_ppg = test_data['samples'].numpy()
        test_co2 = test_data['labels'].numpy()

        # Randomly select indices
        train_indices = random.sample(range(len(train_ppg)), min(num_samples, len(train_ppg)))
        test_indices = random.sample(range(len(test_ppg)), min(num_samples, len(test_ppg)))

        print(f"\nGenerating {len(train_indices)} training sample plots...")
        for idx in tqdm(train_indices, desc="Train Samples"):
            ppg = train_ppg[idx]
            co2 = train_co2[idx]
            save_path = os.path.join(self.waveforms_dir, f'train_sample_{idx}.png')
            self.plot_waveform(ppg, co2, save_path, title_prefix=f"Train Sample {idx}")

        print(f"\nGenerating {len(test_indices)} testing sample plots...")
        for idx in tqdm(test_indices, desc="Test Samples"):
            ppg = test_ppg[idx]
            co2 = test_co2[idx]
            save_path = os.path.join(self.waveforms_dir, f'test_sample_{idx}.png')
            self.plot_waveform(ppg, co2, save_path, title_prefix=f"Test Sample {idx}")

        print(f"\nWaveform plots saved to: {self.waveforms_dir}/")

    def calculate_respiratory_rate(self, co2_signal):
        """
        Calculate respiratory rate (BPM) from CO2 signal.

        Args:
            co2_signal: CO2 signal array

        Returns:
            bpm: Breaths per minute
        """
        # Find peaks in CO2 waveform (each peak represents a breath)
        # Use a minimum distance between peaks (assume minimum RR of 6 BPM -> max 10 breaths in 9.6s)
        min_distance = int(self.target_fs * 60 / 70)  # Assuming max 70 breaths per minute

        peaks, _ = find_peaks(co2_signal, distance=min_distance, prominence=0.1)

        num_peaks = len(peaks)

        # Calculate BPM: (number of breaths / window_duration) * 60
        bpm = (num_peaks / self.window_duration) * 60

        return bpm

    def analyze_respiratory_rate_distribution(self, train_data, test_data):
        """
        Analyze and visualize respiratory rate distribution.

        Args:
            train_data: Training dataset dictionary
            test_data: Testing dataset dictionary
        """
        print("\n" + "=" * 60)
        print("Analyzing Respiratory Rate Distribution")
        print("=" * 60)

        # Get CO2 labels as numpy arrays
        train_co2 = train_data['labels'].numpy()
        test_co2 = test_data['labels'].numpy()

        # Calculate BPM for all training samples
        print("\nCalculating respiratory rates for training set...")
        train_bpm = []
        for i in tqdm(range(len(train_co2)), desc="Train RR Calculation"):
            bpm = self.calculate_respiratory_rate(train_co2[i])
            train_bpm.append(bpm)

        # Calculate BPM for all testing samples
        print("\nCalculating respiratory rates for testing set...")
        test_bpm = []
        for i in tqdm(range(len(test_co2)), desc="Test RR Calculation"):
            bpm = self.calculate_respiratory_rate(test_co2[i])
            test_bpm.append(bpm)

        # Convert to numpy arrays
        train_bpm = np.array(train_bpm)
        test_bpm = np.array(test_bpm)

        # Print statistics
        print("\n" + "-" * 60)
        print("Respiratory Rate Statistics")
        print("-" * 60)
        print(f"\nTraining Set:")
        print(f"  Mean RR: {train_bpm.mean():.2f} BPM")
        print(f"  Std RR:  {train_bpm.std():.2f} BPM")
        print(f"  Min RR:  {train_bpm.min():.2f} BPM")
        print(f"  Max RR:  {train_bpm.max():.2f} BPM")

        print(f"\nTesting Set:")
        print(f"  Mean RR: {test_bpm.mean():.2f} BPM")
        print(f"  Std RR:  {test_bpm.std():.2f} BPM")
        print(f"  Min RR:  {test_bpm.min():.2f} BPM")
        print(f"  Max RR:  {test_bpm.max():.2f} BPM")

        # Create histogram
        print("\nGenerating respiratory rate distribution plot...")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot overlapping histograms
        bins = np.arange(0, max(train_bpm.max(), test_bpm.max()) + 2, 1)

        ax.hist(train_bpm, bins=bins, alpha=0.6, color='blue',
                label=f'Train (n={len(train_bpm)}, μ={train_bpm.mean():.1f})',
                edgecolor='black', linewidth=0.5)
        ax.hist(test_bpm, bins=bins, alpha=0.6, color='red',
                label=f'Test (n={len(test_bpm)}, μ={test_bpm.mean():.1f})',
                edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Breaths Per Minute (BPM)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax.set_title('Respiratory Rate Distribution: Train vs Test',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'rr_distribution.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"Distribution plot saved to: {save_path}")

        return train_bpm, test_bpm

    def run_full_eda(self, num_waveform_samples=50):
        """
        Run complete EDA pipeline.

        Args:
            num_waveform_samples: Number of waveform samples to plot per dataset
        """
        print("\n")
        print("=" * 60)
        print("     DEEP CORR-ENCODER DATASET EDA")
        print("=" * 60)

        # Load datasets
        train_data, test_data = self.load_datasets()

        # Generate waveform gallery
        self.visualize_waveform_gallery(train_data, test_data,
                                       num_samples=num_waveform_samples)

        # Analyze respiratory rate distribution
        self.analyze_respiratory_rate_distribution(train_data, test_data)

        print("\n" + "=" * 60)
        print("EDA Complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}/")
        print(f"  - Waveform plots: {self.waveforms_dir}/")
        print(f"  - RR distribution: {os.path.join(self.output_dir, 'rr_distribution.png')}")
        print("=" * 60)


if __name__ == "__main__":
    os.chdir(ROOT)

    # Initialize EDA analyzer
    eda = DatasetEDA(
        processed_data_dir='processed_data',
        output_dir='eda_results'
    )

    # Run full EDA with 50 samples per dataset
    eda.run_full_eda(num_waveform_samples=50)
