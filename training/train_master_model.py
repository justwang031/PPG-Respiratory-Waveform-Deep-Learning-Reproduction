"""
Deep Corr-Encoder Master Model Training Script

This script trains a "master model" on 100% of the CapnoBase dataset (all 42 subjects)
for use as a pre-trained base model for transfer learning on the BIDMC dataset.

Key differences from regular training:
- Uses ALL 42 subjects (no train/test split)
- No validation loop (100% data used for training)
- Runs for fixed 80 epochs (no early stopping)
- Saves to capnobase_master.pth

Author: Zhantao Wang
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from core_model.utils import CorrEncoder, count_parameters
from preprocessing.preprocess import CapnoBasePreprocessor


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 80

# Paths
MODEL_PATH = 'capnobase_master.pth'
LOG_PATH = 'master_training_log.txt'


# ============================================================================
# Device Detection
# ============================================================================

def get_device():
    """
    Auto-detect the best available device for training.
    Priority: MPS (Apple Silicon GPU) > CUDA > CPU

    Returns:
        torch.device: The selected device
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print(f"Using MPS (Apple Silicon GPU) device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device


# ============================================================================
# Reproducibility
# ============================================================================

def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Loading
# ============================================================================

def load_all_capnobase_data():
    """
    Load 100% of CapnoBase data for master model pre-training.

    This function loads ALL 42 subjects from the raw data directory,
    processes them, and creates a single DataLoader containing all samples.

    Returns:
        tuple: (train_loader, total_samples)
            - train_loader: DataLoader with all data
            - total_samples: Total number of windows loaded
    """
    print("=" * 80)
    print("Loading 100% of CapnoBase Data for Master Model Pre-training")
    print("=" * 80)
    print()

    # Initialize preprocessor
    preprocessor = CapnoBasePreprocessor(
        raw_data_dir='raw_data',
        processed_data_dir='processed_data'
    )

    # Discover all subjects
    subject_ids = preprocessor.discover_subjects()
    print(f"\nLoading ALL {len(subject_ids)} subjects for master training...")
    print("(No train/test split - using 100% of data)")
    print()

    # Accumulate all windows from all subjects
    all_ppg = []
    all_co2 = []

    for subject_id in subject_ids:
        ppg_windows, co2_windows = preprocessor.process_subject(subject_id)
        all_ppg.extend(ppg_windows)
        all_co2.extend(co2_windows)

    # Convert to tensors [N, 288] -> [N, 1, 288]
    ppg_tensor = torch.FloatTensor(all_ppg).unsqueeze(1)
    co2_tensor = torch.FloatTensor(all_co2).unsqueeze(1)

    print()
    print("=" * 80)
    print("Data Loading Summary")
    print("=" * 80)
    print(f"Total windows loaded: {len(all_ppg)}")
    print(f"PPG shape: {ppg_tensor.shape}")
    print(f"CO2 shape: {co2_tensor.shape}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Create dataset and loader
    dataset = TensorDataset(ppg_tensor, co2_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Total batches per epoch: {len(loader)}")

    # Verification: Print first batch shapes
    first_batch = next(iter(loader))
    print(f"First batch - Inputs: {first_batch[0].shape}, Targets: {first_batch[1].shape}")
    print("Expected: torch.Size([32, 1, 288]) for both")
    print()

    return loader, len(all_ppg)


# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The CorrEncoder model
        train_loader: DataLoader for training data
        criterion: Loss function (MSE)
        optimizer: Optimizer (Adam)
        device: Device to train on

    Returns:
        float: Average training loss for the epoch
    """
    model.train()  # Enable dropout
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, targets in progress_bar:
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# ============================================================================
# Logging Functions
# ============================================================================

def initialize_log(log_path, device, model, total_samples):
    """
    Initialize the training log file with header information.

    Args:
        log_path: Path to the log file
        device: Device being used for training
        model: The CorrEncoder model
        total_samples: Total number of training samples
    """
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Deep Corr-Encoder Master Model Training Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: CorrEncoder\n")
        f.write(f"Trainable Parameters: {count_parameters(model):,}\n\n")
        f.write("Training Configuration:\n")
        f.write("  Purpose: Pre-training master model on 100% CapnoBase data\n")
        f.write("  No train/test split - using ALL data for maximum learning\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"  Loss Function: MSE\n")
        f.write(f"  Optimizer: Adam\n\n")
        f.write("Dataset:\n")
        f.write(f"  Total Training Samples: {total_samples}\n")
        f.write(f"  Validation Samples: N/A (using 100% for training)\n\n")
        f.write("=" * 80 + "\n")
        f.write("Training Progress:\n")
        f.write("=" * 80 + "\n\n")


def log_epoch(log_path, epoch, train_loss):
    """
    Append epoch results to the log file.

    Args:
        log_path: Path to the log file
        epoch: Current epoch number (0-indexed)
        train_loss: Training loss for this epoch
    """
    with open(log_path, 'a') as f:
        f.write(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.6f}\n")


def log_summary(log_path, total_time):
    """
    Write final training summary to the log file.

    Args:
        log_path: Path to the log file
        total_time: Total training time in seconds
    """
    with open(log_path, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Completed all {NUM_EPOCHS} epochs\n")
        f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
        f.write(f"Model saved to: {MODEL_PATH}\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """
    Main training function for the master model.

    Trains CorrEncoder on 100% of CapnoBase data (all 42 subjects) for 80 epochs.
    No validation is performed since we use all available data for training.
    """
    # Set random seeds for reproducibility
    seed_everything(seed=42)

    print("=" * 80)
    print("Deep Corr-Encoder Master Model Training")
    print("=" * 80)
    print()

    # Setup device
    device = get_device()
    print()

    # Load ALL data (100% of CapnoBase)
    train_loader, total_samples = load_all_capnobase_data()

    # Initialize model
    model = CorrEncoder().to(device)
    print("=" * 80)
    print(f"Model initialized with {count_parameters(model):,} parameters")
    print("=" * 80)
    print()

    # Loss and optimizer (from paper specification)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize logging
    initialize_log(LOG_PATH, device, model, total_samples)

    # Training state
    train_losses = []

    # Start training
    start_time = time.time()

    print("=" * 80)
    print("Starting Training (80 epochs, no early stopping)")
    print("=" * 80)
    print()

    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Save loss
        train_losses.append(train_loss)

        # Log epoch
        log_epoch(LOG_PATH, epoch, train_loss)

        # Print progress
        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.6f}")

    # Training complete - save final model
    print()
    print("=" * 80)
    print("Saving Final Model")
    print("=" * 80)

    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
    }, MODEL_PATH)

    total_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print()

    # Log summary
    log_summary(LOG_PATH, total_time)

    print(f"Master pre-trained model saved to: {MODEL_PATH}")
    print(f"Training log saved to: {LOG_PATH}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
