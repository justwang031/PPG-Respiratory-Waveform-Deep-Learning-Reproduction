"""
Deep Corr-Encoder Model Visualization Script

This script loads the trained model and creates visualizations of its predictions
on random test samples, comparing PPG input, ground truth CO2, and predicted CO2.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import CorrEncoder


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = 'best_model.pth'
DATA_PATH = 'processed_data/test.pt'
OUTPUT_PATH = 'prediction_gallery.png'
NUM_SAMPLES = 10


# ============================================================================
# Device Detection
# ============================================================================

def get_device():
    """
    Auto-detect the best available device for inference.
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
# Model Loading
# ============================================================================

def load_model(model_path, device):
    """
    Load the trained CorrEncoder model from checkpoint.

    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        device: Device to load the model onto

    Returns:
        model: Loaded CorrEncoder model in eval mode
    """
    # Initialize model architecture
    model = CorrEncoder().to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode (disables dropout)
    model.eval()

    print(f"Model loaded from: {model_path}")
    print(f"Best epoch: {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")

    return model


# ============================================================================
# Data Loading
# ============================================================================

def load_test_data(data_path):
    """
    Load the test dataset.

    Args:
        data_path: Path to the test data (.pt file)

    Returns:
        tuple: (samples, labels) both with shape [N, 1, 288]
    """
    # Load test data
    test_data = torch.load(data_path)
    samples = test_data['samples'].unsqueeze(1)  # [150, 288] -> [150, 1, 288]
    labels = test_data['labels'].unsqueeze(1)    # [150, 288] -> [150, 1, 288]

    print(f"Test data loaded from: {data_path}")
    print(f"Test samples shape: {samples.shape}")
    print(f"Test labels shape: {labels.shape}")

    return samples, labels


# ============================================================================
# Inference
# ============================================================================

def run_inference(model, samples, device):
    """
    Run inference on the test samples.

    Args:
        model: Trained CorrEncoder model
        samples: Input PPG signals [N, 1, 288]
        device: Device to run inference on

    Returns:
        predictions: Model predictions [N, 1, 288]
    """
    with torch.no_grad():
        samples = samples.to(device)
        predictions = model(samples)

    return predictions


# ============================================================================
# Visualization
# ============================================================================

def visualize_predictions(inputs, targets, predictions, save_path, num_samples=5):
    """
    Create a gallery of prediction visualizations.

    For each sample, plots:
    - Blue: Input PPG (for reference)
    - Orange: Ground Truth CO2
    - Green (Dashed): Model Predicted CO2

    Args:
        inputs: Input PPG signals [N, 1, 288]
        targets: Ground truth CO2 signals [N, 1, 288]
        predictions: Model predicted CO2 signals [N, 1, 288]
        save_path: Path to save the visualization
        num_samples: Number of random samples to visualize
    """
    # Move tensors to CPU and convert to numpy
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Randomly select samples
    num_total = inputs.shape[0]
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(num_total, size=num_samples, replace=False)
    selected_indices = sorted(selected_indices)

    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))

    # Handle case where num_samples = 1
    if num_samples == 1:
        axes = [axes]

    # Time axis (0 to 287)
    time_axis = np.arange(288)

    # Plot each sample
    for i, idx in enumerate(selected_indices):
        ax = axes[i]

        # Extract data (remove channel dimension)
        ppg = inputs[idx, 0, :]        # [288]
        gt_co2 = targets[idx, 0, :]    # [288]
        pred_co2 = predictions[idx, 0, :]  # [288]

        # Calculate MSE loss for this sample
        mse_loss = np.mean((gt_co2 - pred_co2) ** 2)

        # Plot three lines
        ax.plot(time_axis, ppg, 'b-', linewidth=1.5, alpha=0.6, label='Input PPG')
        ax.plot(time_axis, gt_co2, color='orange', linewidth=2, label='Ground Truth CO2')
        ax.plot(time_axis, pred_co2, 'g--', linewidth=2, label='Predicted CO2')

        # Formatting
        ax.set_title(f'Sample {idx} | MSE Loss: {mse_loss:.6f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Normalized Signal', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 287)

    # Overall title
    fig.suptitle('Deep Corr-Encoder: Prediction Gallery',
                 fontsize=16, fontweight='bold', y=0.995)

    # Layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.995])

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Prediction gallery saved to: {save_path}")
    print(f"Visualized samples: {selected_indices}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function that orchestrates the visualization pipeline.
    """
    print("=" * 80)
    print("Deep Corr-Encoder Model Visualization")
    print("=" * 80)
    print()

    # Setup device
    device = get_device()
    print()

    # Load model
    model = load_model(MODEL_PATH, device)
    print()

    # Load test data
    test_samples, test_labels = load_test_data(DATA_PATH)
    print()

    # Run inference
    print("Running inference...")
    predictions = run_inference(model, test_samples, device)
    print(f"Predictions shape: {predictions.shape}")

    # Calculate overall test loss
    criterion = nn.MSELoss()
    test_loss = criterion(predictions, test_labels.to(device)).item()
    print(f"Overall test MSE: {test_loss:.6f}")
    print()

    # Create visualization
    print(f"Creating visualization for {NUM_SAMPLES} random samples...")
    visualize_predictions(
        inputs=test_samples,
        targets=test_labels,
        predictions=predictions,
        save_path=OUTPUT_PATH,
        num_samples=NUM_SAMPLES
    )
    print()

    print("=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
