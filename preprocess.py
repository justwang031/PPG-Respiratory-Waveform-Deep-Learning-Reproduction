"""
CapnoBase Dataset Preprocessing Script for Deep Corr-Encoder

This script processes raw CapnoBase physiological signals for deep learning.
It performs resampling, filtering, normalization, and segmentation.

Author: Zhantao Wang
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from scipy import signal
from pathlib import Path


class CapnoBasePreprocessor:
    """Preprocessor for CapnoBase dataset"""

    def __init__(self, raw_data_dir='raw_data', processed_data_dir='processed_data'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.target_fs = 30  # Target sampling rate in Hz
        self.window_duration = 9.6  # Window duration in seconds
        self.window_samples = int(self.target_fs * self.window_duration)  # 288 samples

        # Create output directory if it doesn't exist
        Path(self.processed_data_dir).mkdir(parents=True, exist_ok=True)

    def discover_subjects(self):
        """
        Discover and match signal/param file pairs.
        Returns sorted list of subject IDs.
        """
        signal_files = glob.glob(os.path.join(self.raw_data_dir, '*_signal.csv'))
        subject_ids = []

        for signal_file in signal_files:
            # Extract subject ID (e.g., '0009_8min' from '0009_8min_signal.csv')
            basename = os.path.basename(signal_file)
            subject_id = basename.replace('_signal.csv', '')

            # Check if corresponding param file exists
            param_file = os.path.join(self.raw_data_dir, f'{subject_id}_param.csv')
            if os.path.exists(param_file):
                subject_ids.append(subject_id)

        # Sort alphabetically for reproducible splitting
        subject_ids.sort()

        print(f"Found {len(subject_ids)} subjects with complete data pairs")
        return subject_ids

    def load_subject_data(self, subject_id):
        """
        Load signal and parameter data for a subject.

        Returns:
            ppg: PPG signal (pleth_y)
            co2: CO2 signal (co2_y)
            original_fs: Original sampling rate
        """
        signal_file = os.path.join(self.raw_data_dir, f'{subject_id}_signal.csv')
        param_file = os.path.join(self.raw_data_dir, f'{subject_id}_param.csv')

        # Load signal data
        signal_df = pd.read_csv(signal_file)
        ppg = signal_df['pleth_y'].values
        co2 = signal_df['co2_y'].values

        # Load parameter data
        param_df = pd.read_csv(param_file)
        original_fs = param_df['samplingrate_pleth'].values[0]

        return ppg, co2, original_fs

    def resample_signal(self, data, original_fs, target_fs):
        """
        Resample signal from original_fs to target_fs.

        Args:
            data: Input signal
            original_fs: Original sampling rate
            target_fs: Target sampling rate

        Returns:
            Resampled signal
        """
        if original_fs == target_fs:
            return data

        # Calculate number of samples after resampling
        num_samples = int(len(data) * target_fs / original_fs)

        # Use scipy's resample function
        resampled = signal.resample(data, num_samples)

        return resampled

    def apply_highpass_filter(self, data, fs, cutoff=0.05, order=2):
        """
        Apply 2nd-order Butterworth high-pass filter to remove DC baseline wander.

        Args:
            data: Input signal
            fs: Sampling rate
            cutoff: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered signal
        """
        # Design Butterworth high-pass filter
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)

        # Apply filter
        filtered = signal.filtfilt(b, a, data)

        return filtered

    def normalize_signal(self, data, range_min, range_max):
        """
        Normalize signal to specified range.

        Args:
            data: Input signal
            range_min: Minimum value of output range
            range_max: Maximum value of output range

        Returns:
            Normalized signal
        """
        data_min = np.min(data)
        data_max = np.max(data)

        # Avoid division by zero
        if data_max - data_min == 0:
            return np.full_like(data, (range_min + range_max) / 2)

        # Min-max normalization to [range_min, range_max]
        normalized = (data - data_min) / (data_max - data_min)
        normalized = normalized * (range_max - range_min) + range_min

        return normalized

    def segment_signal(self, ppg, co2):
        """
        Segment signals into non-overlapping windows.

        Args:
            ppg: PPG signal
            co2: CO2 signal

        Returns:
            ppg_windows: List of PPG windows
            co2_windows: List of CO2 windows
        """
        num_samples = len(ppg)
        num_windows = num_samples // self.window_samples

        ppg_windows = []
        co2_windows = []

        for i in range(num_windows):
            start_idx = i * self.window_samples
            end_idx = start_idx + self.window_samples

            ppg_windows.append(ppg[start_idx:end_idx])
            co2_windows.append(co2[start_idx:end_idx])

        return ppg_windows, co2_windows

    def process_subject(self, subject_id):
        """
        Complete preprocessing pipeline for one subject.

        Args:
            subject_id: Subject identifier

        Returns:
            ppg_windows: List of preprocessed PPG windows
            co2_windows: List of preprocessed CO2 windows
        """
        print(f"Processing subject {subject_id}...")

        # Load data
        ppg, co2, original_fs = self.load_subject_data(subject_id)

        # Resample to 30 Hz
        ppg_resampled = self.resample_signal(ppg, original_fs, self.target_fs)
        co2_resampled = self.resample_signal(co2, original_fs, self.target_fs)

        # Apply high-pass filter to PPG only
        ppg_filtered = self.apply_highpass_filter(ppg_resampled, self.target_fs, cutoff=0.05, order=2)

        # Normalize signals
        ppg_normalized = self.normalize_signal(ppg_filtered, -1, 1)  # PPG to [-1, 1]
        co2_normalized = self.normalize_signal(co2_resampled, 0, 1)  # CO2 to [0, 1]

        # Segment into windows
        ppg_windows, co2_windows = self.segment_signal(ppg_normalized, co2_normalized)

        print(f"  Created {len(ppg_windows)} windows of {self.window_samples} samples each")

        return ppg_windows, co2_windows

    def process_dataset(self):
        """
        Process entire dataset and save to .pt files.
        """
        print("=" * 60)
        print("CapnoBase Dataset Preprocessing")
        print("=" * 60)

        # Discover all subjects
        subject_ids = self.discover_subjects()

        # Split: first 39 for training, last 3 for testing
        train_subjects = subject_ids[:39]
        test_subjects = subject_ids[-3:]

        print(f"\nTraining subjects: {len(train_subjects)}")
        print(f"Testing subjects: {len(test_subjects)}")
        print(f"Test subject IDs: {test_subjects}")

        # Process training data
        print("\n" + "=" * 60)
        print("Processing Training Data")
        print("=" * 60)
        train_ppg_all = []
        train_co2_all = []

        for subject_id in train_subjects:
            ppg_windows, co2_windows = self.process_subject(subject_id)
            train_ppg_all.extend(ppg_windows)
            train_co2_all.extend(co2_windows)

        # Convert to tensors
        train_ppg_tensor = torch.FloatTensor(train_ppg_all)
        train_co2_tensor = torch.FloatTensor(train_co2_all)

        # Save training data
        train_data = {
            'samples': train_ppg_tensor,
            'labels': train_co2_tensor
        }
        train_path = os.path.join(self.processed_data_dir, 'train.pt')
        torch.save(train_data, train_path)
        print(f"\nTraining data saved: {train_path}")
        print(f"  Shape - Samples: {train_ppg_tensor.shape}, Labels: {train_co2_tensor.shape}")

        # Process testing data
        print("\n" + "=" * 60)
        print("Processing Testing Data")
        print("=" * 60)
        test_ppg_all = []
        test_co2_all = []

        for subject_id in test_subjects:
            ppg_windows, co2_windows = self.process_subject(subject_id)
            test_ppg_all.extend(ppg_windows)
            test_co2_all.extend(co2_windows)

        # Convert to tensors
        test_ppg_tensor = torch.FloatTensor(test_ppg_all)
        test_co2_tensor = torch.FloatTensor(test_co2_all)

        # Save testing data
        test_data = {
            'samples': test_ppg_tensor,
            'labels': test_co2_tensor
        }
        test_path = os.path.join(self.processed_data_dir, 'test.pt')
        torch.save(test_data, test_path)
        print(f"\nTesting data saved: {test_path}")
        print(f"  Shape - Samples: {test_ppg_tensor.shape}, Labels: {test_co2_tensor.shape}")

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"Total training windows: {len(train_ppg_all)}")
        print(f"Total testing windows: {len(test_ppg_all)}")
        print(f"Window size: {self.window_samples} samples ({self.window_duration}s @ {self.target_fs}Hz)")


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CapnoBasePreprocessor(
        raw_data_dir='raw_data',
        processed_data_dir='processed_data'
    )

    # Process the dataset
    preprocessor.process_dataset()
