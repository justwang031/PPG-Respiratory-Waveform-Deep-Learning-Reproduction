"""
Deep Corr-Encoder Model Definition

This module implements the 1D Convolutional Autoencoder architecture
for PPG to CO2 signal reconstruction as described in the Deep Corr-Encoder paper.
"""

import torch
import torch.nn as nn


class CorrEncoder(nn.Module):
    """
    1D Convolutional Autoencoder for PPG to CO2 signal reconstruction.

    Architecture from Deep Corr-Encoder paper (Fig 1, Section 2.2):
    - Encoder: 3 Conv1d layers with Dropout(0.5) and ReLU/Sigmoid activations
    - Decoder: 3 ConvTranspose1d layers (mirrored, no dropout)

    Dimension flow: 288 → 179 → 125 → 76 → 125 → 179 → 288

    Input:  [N, 1, 288] - PPG signals normalized to [-1, 1]
    Output: [N, 1, 288] - CO2 signals normalized to [0, 1]

    Total parameters: 18,441
    """

    def __init__(self):
        super(CorrEncoder, self).__init__()

        # Encoder with dropout (applied to ALL encoder layers)
        self.encoder = nn.Sequential(
            # Layer 1: 288 -> 179
            # Conv1d: L_out = L_in - kernel_size + 2*padding + 1
            #       = 288 - 150 + 40 + 1 = 179
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=150, padding=20),
            nn.Dropout(0.5),
            nn.ReLU(),

            # Layer 2: 179 -> 125
            # Conv1d: 179 - 75 + 20 + 1 = 125
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=75, padding=10),
            nn.Dropout(0.5),
            nn.ReLU(),

            # Layer 3: 125 -> 76 (bottleneck)
            # Conv1d: 125 - 50 + 0 + 1 = 76
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=50, padding=0),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        # Decoder without dropout (mirrored structure)
        self.decoder = nn.Sequential(
            # Layer 4: 76 -> 125
            # ConvTranspose1d: L_out = (L_in - 1) + kernel_size - 2*padding
            #                = (76 - 1) + 50 - 0 = 125
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=50, padding=0),
            nn.Sigmoid(),

            # Layer 5: 125 -> 179
            # ConvTranspose1d: (125 - 1) + 75 - 20 = 179
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=75, padding=10),
            nn.ReLU(),

            # Layer 6: 179 -> 288 (output layer)
            # ConvTranspose1d: (179 - 1) + 150 - 40 = 288
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=150, padding=20),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor [N, 1, 288] - PPG signals

        Returns:
            Reconstructed signal [N, 1, 288] - CO2 signals
        """
        # Expects [N, 1, 288], returns [N, 1, 288]
        encoded = self.encoder(x)  # [N, 1, 288] -> [N, 8, 76]
        decoded = self.decoder(encoded)  # [N, 8, 76] -> [N, 1, 288]
        return decoded


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return total_params


if __name__ == "__main__":
    # Test the model
    print("Testing CorrEncoder Model")
    print("=" * 60)

    # Instantiate model
    model = CorrEncoder()
    print("✓ Model instantiated successfully")

    # Count parameters
    param_count = count_parameters(model)
    assert param_count == 18441, f"Expected 18,441 parameters, got {param_count}"
    print("✓ Parameter count verified: 18,441")

    # Test forward pass with correct dimensions
    test_input = torch.randn(32, 1, 288)
    print(f"\nTest input shape: {test_input.shape}")

    model.eval()
    with torch.no_grad():
        test_output = model(test_input)

    print(f"Test output shape: {test_output.shape}")
    assert test_output.shape == (32, 1, 288), f"Expected shape (32, 1, 288), got {test_output.shape}"
    print("✓ Forward pass successful: [32, 1, 288] -> [32, 1, 288]")

    print("\n" + "=" * 60)
    print("All tests passed! Model is ready for training.")
