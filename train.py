"""
Deep Corr-Encoder Training Script

This script implements the complete training pipeline for the Deep Corr-Encoder model,
including data loading, training loop, validation, logging, and visualization.
"""

import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import CorrEncoder, count_parameters


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 80

# Paths
DATA_DIR = 'processed_data'
MODEL_PATH = 'best_model.pth'
LOG_PATH = 'training_log.txt'
PLOT_PATH = 'loss_curve.png'


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
# Data Loading
# ============================================================================

def load_data():
    """
    Load training and validation datasets from .pt files.

    Returns:
        tuple: (train_loader, val_loader, train_size, val_size)
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - train_size: Number of training samples
            - val_size: Number of validation samples
    """
    # Load training data
    train_data = torch.load(os.path.join(DATA_DIR, 'train.pt'))
    train_samples = train_data['samples'].unsqueeze(1)  # [1950, 288] -> [1950, 1, 288]
    train_labels = train_data['labels'].unsqueeze(1)    # [1950, 288] -> [1950, 1, 288]

    # Load validation data (test.pt used as validation set)
    val_data = torch.load(os.path.join(DATA_DIR, 'test.pt'))
    val_samples = val_data['samples'].unsqueeze(1)  # [150, 288] -> [150, 1, 288]
    val_labels = val_data['labels'].unsqueeze(1)    # [150, 288] -> [150, 1, 288]

    # Create TensorDatasets
    train_dataset = TensorDataset(train_samples, train_labels)
    val_dataset = TensorDataset(val_samples, val_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Safety check: Print first batch shapes
    first_batch = next(iter(train_loader))
    print(f"\nFirst batch - Inputs: {first_batch[0].shape}, Targets: {first_batch[1].shape}")
    print("Expected: torch.Size([32, 1, 288]) for both\n")

    return train_loader, val_loader, len(train_samples), len(val_samples)


# ============================================================================
# Training and Validation Functions
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


def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Args:
        model: The CorrEncoder model
        val_loader: DataLoader for validation data
        criterion: Loss function (MSE)
        device: Device to validate on

    Returns:
        float: Average validation loss
    """
    model.eval()  # Disable dropout
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(val_loader)
    return avg_loss


# ============================================================================
# Logging Functions
# ============================================================================

def initialize_log(log_path, device, model, train_size, val_size):
    """
    Initialize the training log file with header information.

    Args:
        log_path: Path to the log file
        device: Device being used for training
        model: The CorrEncoder model
        train_size: Number of training samples
        val_size: Number of validation samples
    """
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Deep Corr-Encoder Training Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model: CorrEncoder\n")
        f.write(f"Trainable Parameters: {count_parameters(model):,}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"  Loss Function: MSE\n")
        f.write(f"  Optimizer: Adam\n\n")
        f.write("Dataset:\n")
        f.write(f"  Training Samples: {train_size}\n")
        f.write(f"  Validation Samples: {val_size}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Training Progress:\n")
        f.write("=" * 80 + "\n\n")


def log_epoch(log_path, epoch, train_loss, val_loss, is_best):
    """
    Append epoch results to the log file.

    Args:
        log_path: Path to the log file
        epoch: Current epoch number (0-indexed)
        train_loss: Training loss for this epoch
        val_loss: Validation loss for this epoch
        is_best: Whether this is the best model so far
    """
    with open(log_path, 'a') as f:
        best_marker = " [BEST MODEL SAVED]" if is_best else ""
        f.write(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}{best_marker}\n")


def log_summary(log_path, best_epoch, best_val_loss, total_time):
    """
    Write final training summary to the log file.

    Args:
        log_path: Path to the log file
        best_epoch: Epoch number where best model was found (0-indexed)
        best_val_loss: Best validation loss achieved
        total_time: Total training time in seconds
    """
    with open(log_path, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
        f.write(f"Model saved to: {MODEL_PATH}\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# Visualization
# ============================================================================

def plot_losses(train_losses, val_losses, save_path):
    """
    Create and save the training/validation loss curve plot.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Deep Corr-Encoder Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")


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
# Main Training Loop
# ============================================================================

def main():
    """
    Main training function that orchestrates the entire training pipeline.
    """
    # Set random seeds for reproducibility
    seed_everything(seed=42)

    print("=" * 80)
    print("Deep Corr-Encoder Training")
    print("=" * 80)
    print()

    # Setup device
    device = get_device()
    print()

    # Load data
    train_loader, val_loader, train_size, val_size = load_data()
    print()

    # Initialize model
    model = CorrEncoder().to(device)
    print(f"Model initialized with {count_parameters(model):,} parameters")
    print()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize logging
    initialize_log(LOG_PATH, device, model, train_size, val_size)

    # Training state
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    # Start training
    start_time = time.time()

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check if best model
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
            }, MODEL_PATH)

        # Log epoch
        log_epoch(LOG_PATH, epoch, train_loss, val_loss, is_best)

        # Print progress
        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}" +
              (" [BEST]" if is_best else ""))

    # Training complete
    total_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print()

    # Log summary
    log_summary(LOG_PATH, best_epoch, best_val_loss, total_time)

    # Plot losses
    plot_losses(train_losses, val_losses, PLOT_PATH)

    print(f"Training log saved to: {LOG_PATH}")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
