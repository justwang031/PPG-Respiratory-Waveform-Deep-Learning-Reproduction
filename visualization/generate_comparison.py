"""
Comparison Plot Generator for Deep Corr-Encoder vs PLS Baseline
================================================================

This script generates a box plot comparing waveform reconstruction error (MAE)
between the Deep Corr-Encoder model and the PLS baseline.

Requirements:
    - loso_results/all_fold_results.csv (Deep Learning results)
    - pls_results/pls_all_fold_results.csv (PLS baseline results)

Output:
    - pls_results/waveform_mae_comparison.png
"""

import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    os.chdir(ROOT)

    # File paths
    dl_results_path = "loso_results/all_fold_results.csv"
    pls_results_path = "pls_results/pls_all_fold_results.csv"

    # Check if files exist
    if not Path(dl_results_path).exists():
        raise FileNotFoundError(
            f"Deep Learning results file not found: {dl_results_path}\n"
            "Please run loso_eval.py first to generate the results."
        )
    if not Path(pls_results_path).exists():
        raise FileNotFoundError(
            f"PLS results file not found: {pls_results_path}\n"
            "Please run run_pls_baseline.py first to generate the results."
        )

    # Load the data
    print("Loading results...")
    dl_df = pd.read_csv(dl_results_path)
    pls_df = pd.read_csv(pls_results_path)

    # Extract waveform MAE
    dl_mae = dl_df['waveform_mae'].values
    pls_mae = pls_df['waveform_mae'].values

    # Print statistics
    print("\n" + "="*70)
    print("WAVEFORM RECONSTRUCTION ERROR STATISTICS")
    print("="*70)
    print(f"\nDeep Corr-Encoder (n={len(dl_mae)} folds):")
    print(f"  Median MAE: {np.median(dl_mae):.6f}")
    print(f"  Mean MAE:   {np.mean(dl_mae):.6f} ± {np.std(dl_mae):.6f}")

    print(f"\nPLS Baseline (n={len(pls_mae)} folds):")
    print(f"  Median MAE: {np.median(pls_mae):.6f}")
    print(f"  Mean MAE:   {np.mean(pls_mae):.6f} ± {np.std(pls_mae):.6f}")

    # Calculate improvement
    improvement = ((np.mean(pls_mae) - np.mean(dl_mae)) / np.mean(pls_mae)) * 100
    print(f"\nRelative Improvement: {improvement:.2f}%")
    print("="*70 + "\n")

    # Create the box plot
    print("Generating comparison plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create box plot
    bp = ax.boxplot(
        [dl_mae, pls_mae],
        labels=['Deep Corr-Encoder', 'PLS Baseline'],
        patch_artist=True,
        showmeans=False,
        widths=0.6,
        medianprops=dict(color='red', linewidth=2.5),
        flierprops=dict(
            marker='o',
            markerfacecolor='gray',
            markersize=6,
            alpha=0.5,
            markeredgecolor='darkgray'
        ),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    # Set colors
    colors = ['#87CEEB', '#D3D3D3']  # Light Blue for DL, Light Grey for PLS
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    # Labels and title
    ax.set_ylabel('Mean Absolute Waveform Error (MAE)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_title('Waveform Reconstruction Error Comparison',
                 fontsize=15, fontweight='bold', pad=15)

    # Grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    output_path = "pls_results/waveform_mae_comparison.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")

    plt.close()
    print("Done!\n")


if __name__ == "__main__":
    main()
