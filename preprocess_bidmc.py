"""
BIDMC Dataset Preprocessing Script for Deep Corr-Encoder

This script processes raw BIDMC physiological signals for deep learning.
It performs resampling, filtering, normalization, and segmentation.

Dataset: BIDMC (53 subjects)
- Input signal: PLETH (photoplethysmogram/PPG)
- Label signal: RESP (respiratory/impedance)
- Sampling rate: 125 Hz (fixed)
- Output: Single combined file with all subjects

Author: Adapted from CapnoBase preprocessing
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from scipy import signal
from pathlib import Path


class BIDMCPreprocessor:
    """Preprocessor for BIDMC dataset"""

    def __init__(self, raw_data_dir='raw_data_bidmc', processed_data_dir='processed_data_bidmc'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.original_fs = 125  # BIDMC has fixed sampling rate of 125 Hz
        self.target_fs = 30  # Target sampling rate in Hz
        self.window_duration = 9.6  # Window duration in seconds
        self.window_samples = int(self.target_fs * self.window_duration)  # 288 samples

        # Create output directory if it doesn't exist
        Path(self.processed_data_dir).mkdir(parents=True, exist_ok=True)

    def discover_subjects(self):
        """
        Discover BIDMC signal files and extract subject IDs.
        Returns sorted list of subject IDs.
        """
        signal_files = glob.glob(os.path.join(self.raw_data_dir, 'bidmc_*_Signals.csv'))
        subject_ids = []

        for signal_file in signal_files:
            # Extract subject ID (e.g., '01' from 'bidmc_01_Signals.csv')
            basename = os.path.basename(signal_file)
            subject_id = basename.replace('_Signals.csv', '').replace('bidmc_', '')
            subject_ids.append(subject_id)

        # Sort numerically for consistent ordering
        subject_ids.sort(key=lambda x: int(x))

        print(f"Found {len(subject_ids)} subjects with signal files")
        return subject_ids

    def load_subject_data(self, subject_id):
        """
        Load PLETH and RESP signals for a subject.

        BIDMC Dataset Mapping:
        - PLETH column → PPG input signal
        - RESP column → Respiratory label signal

        Returns:
            pleth: PLETH/PPG signal
            resp: RESP/respiratory signal
        """
        signal_file = os.path.join(self.raw_data_dir, f'bidmc_{subject_id}_Signals.csv')

        # Load signal data and strip whitespace from column names
        signal_df = pd.read_csv(signal_file)
        signal_df.columns = signal_df.columns.str.strip()

        pleth = signal_df['PLETH'].values
        resp = signal_df['RESP'].values

        # Note: BIDMC has fixed 125 Hz sampling rate (no param file needed)
        return pleth, resp

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

    def segment_signal(self, pleth, resp):
        """
        Segment signals into non-overlapping windows.

        Args:
            pleth: PLETH/PPG signal
            resp: RESP/respiratory signal

        Returns:
            pleth_windows: List of PLETH windows
            resp_windows: List of RESP windows
        """
        num_samples = len(pleth)
        num_windows = num_samples // self.window_samples

        pleth_windows = []
        resp_windows = []

        for i in range(num_windows):
            start_idx = i * self.window_samples
            end_idx = start_idx + self.window_samples

            pleth_windows.append(pleth[start_idx:end_idx])
            resp_windows.append(resp[start_idx:end_idx])

        return pleth_windows, resp_windows

    def process_subject(self, subject_id):
        """
        Complete preprocessing pipeline for one subject.

        Args:
            subject_id: Subject identifier

        Returns:
            pleth_windows: List of preprocessed PLETH windows
            resp_windows: List of preprocessed RESP windows
        """
        print(f"Processing subject {subject_id}...")

        # Load data
        pleth, resp = self.load_subject_data(subject_id)

        # Resample to 30 Hz
        pleth_resampled = self.resample_signal(pleth, self.original_fs, self.target_fs)
        resp_resampled = self.resample_signal(resp, self.original_fs, self.target_fs)

        # Apply high-pass filter to PLETH only (input signal)
        # Note: DO NOT filter RESP (label signal) to preserve respiratory waveform
        pleth_filtered = self.apply_highpass_filter(pleth_resampled, self.target_fs, cutoff=0.05, order=2)

        # Normalize signals
        pleth_normalized = self.normalize_signal(pleth_filtered, -1, 1)  # PLETH to [-1, 1]
        resp_normalized = self.normalize_signal(resp_resampled, 0, 1)  # RESP to [0, 1]

        # Segment into windows
        pleth_windows, resp_windows = self.segment_signal(pleth_normalized, resp_normalized)

        print(f"  Created {len(pleth_windows)} windows of {self.window_samples} samples each")

        return pleth_windows, resp_windows

    def process_dataset(self):
        """
        Process entire BIDMC dataset and save to single .pt file.
        All 53 subjects are combined into one output file.
        """
        print("=" * 60)
        print("BIDMC Dataset Preprocessing")
        print("=" * 60)

        # Discover all subjects
        subject_ids = self.discover_subjects()

        print(f"\nProcessing all {len(subject_ids)} subjects")
        print("=" * 60)

        # Process all subjects (no train/test split)
        pleth_all = []
        resp_all = []

        for subject_id in subject_ids:
            pleth_windows, resp_windows = self.process_subject(subject_id)
            pleth_all.extend(pleth_windows)
            resp_all.extend(resp_windows)

        # Convert to tensors
        pleth_tensor = torch.FloatTensor(pleth_all)
        resp_tensor = torch.FloatTensor(resp_all)

        # Save combined dataset
        data = {
            'samples': pleth_tensor,
            'labels': resp_tensor
        }
        output_path = os.path.join(self.processed_data_dir, 'bidmc_all.pt')
        torch.save(data, output_path)

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"Data saved: {output_path}")
        print(f"  Samples shape: {pleth_tensor.shape}")
        print(f"  Labels shape: {resp_tensor.shape}")
        print(f"  Total windows: {len(pleth_all)}")
        print(f"  Window size: {self.window_samples} samples ({self.window_duration}s @ {self.target_fs}Hz)")
        print("=" * 60)


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = BIDMCPreprocessor(
        raw_data_dir='raw_data_bidmc',
        processed_data_dir='processed_data_bidmc'
    )

    # Process the dataset
    preprocessor.process_dataset()
