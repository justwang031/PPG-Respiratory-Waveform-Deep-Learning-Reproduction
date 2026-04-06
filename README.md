# Deep Corr-Encoder: Respiratory Signal Estimation from PPG

**Paper Reproduction with Rigorous Evaluation**

A complete implementation of the Deep Corr-Encoder architecture for non-invasive respiratory monitoring using photoplethysmogram (PPG) signals from pulse oximeters.

---

## Overview

This project reproduces the **Deep Corr-Encoder** model, a 1D convolutional autoencoder that estimates respiratory waveforms (CO2 or respiratory impedance) from photoplethysmogram (PPG) signals. This enables continuous, non-invasive respiratory monitoring using only a pulse oximeter.

**Key Components**:
- **Architecture**: Lightweight 3-layer encoder-decoder CNN (18,441 parameters)
- **Input**: PPG signals from pulse oximeters, normalized to [-1, 1]
- **Output**: Respiratory waveforms (CO2 or RESP), normalized to [0, 1]
- **Window**: 9.6-second segments at 30 Hz (288 samples)

**Datasets**:
- **CapnoBase** (42 subjects): PPG → CO2 waveform prediction
- **BIDMC** (53 subjects): Transfer learning evaluation (PPG → RESP)

**Evaluation Rigor**:
- 42-fold Leave-One-Subject-Out (LOSO) cross-validation
- PLS regression baseline for comparison
- Zero-shot and fine-tuning transfer learning experiments
- Sliding window inference with overlap-add fusion

---

## Key Features

- Complete Deep Corr-Encoder implementation (18,441 parameters verified)
- 42-fold LOSO cross-validation with per-subject evaluation
- PLS regression baseline (26 components) for method comparison
- Zero-shot and fine-tuning transfer learning (CapnoBase → BIDMC)
- Sliding window inference with overlap-add fusion (stride=30, 90% overlap)
- Comprehensive metrics:
  - Waveform MAE (mean absolute error)
  - Respiratory rate error (FFT-based, BPM)
  - Duty cycle error (inhalation fraction)
- Reproducible experiments (seed=42, deterministic operations)
- Publication-quality visualization scripts
- Automatic device detection (MPS/CUDA/CPU)
- Resume support for interrupted LOSO folds

---

## Requirements

- **Python**: >= 3.13
- **Core ML**:
  - `torch` >= 2.9.1
  - `torchvision` >= 0.24.1
  - `scikit-learn` >= 1.5.0
- **Signal Processing**:
  - `scipy` >= 1.16.3
  - `numpy` >= 2.3.5
- **Data & Visualization**:
  - `pandas` >= 2.3.3
  - `matplotlib` >= 3.10.7
  - `seaborn` >= 0.13.2
- **Utilities**:
  - `tqdm` >= 4.67.1
  - `marimo` >= 0.18.1

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Paper_Reproduction
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3.13 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using pip
pip install -e .

# OR using uv (faster)
uv pip install -e .
```

### 4. Verify Installation

```bash
python core_model/utils.py
# Expected output: "All tests passed! Model is ready for training."
```

---

## Data Preparation

### CapnoBase Dataset

**Download**:
- Dataset DOI: [10.5683/SP2/NLB8IT](https://doi.org/10.5683/SP2/NLB8IT)
- Extract files to the `raw_data/` directory

**Expected Structure**:
```
raw_data/
  ├── 0009_8min_signal.csv
  ├── 0009_8min_param.csv
  ├── 0010_8min_signal.csv
  ├── 0010_8min_param.csv
  └── ... (42 subjects total, 84 files)
```

**Preprocessing**:
```bash
python preprocessing/preprocess.py
```

**Output**:
- `processed_data/train.pt` - 39 subjects, ~1950 windows
- `processed_data/test.pt` - 3 subjects, ~150 windows

**Preprocessing Steps** (from `preprocess.py`):
1. Resample signals to 30 Hz (from variable 125-300 Hz)
2. Apply 2nd-order Butterworth high-pass filter to PPG (cutoff=0.05 Hz)
3. Normalize: PPG → [-1, 1], CO2 → [0, 1]
4. Segment into non-overlapping 9.6s windows (288 samples)

---

### BIDMC Dataset (for Transfer Learning)

**Download**:
- Obtain from official dataset source
- Extract to `raw_data_bidmc/` directory

**Expected Structure**:
```
raw_data_bidmc/
  ├── bidmc_01_Signals.csv
  ├── bidmc_02_Signals.csv
  └── ... (53 subjects total)
```

**Preprocessing**:
```bash
python preprocessing/preprocess_bidmc.py
```

**Output**:
- `processed_data_bidmc/bidmc_all.pt` - 53 subjects, ~2650 windows

---

## Usage Guide

### Quick Start: Single Model Training

Train a single model on 39 subjects (validate on 3 subjects):

```bash
python training/train.py
```

**Configuration** (from `train.py`):
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 80
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

**Outputs**:
- `best_model.pth` - Best model checkpoint
- `training_log.txt` - Epoch-by-epoch metrics
- `loss_curve.png` - Training/validation loss curves

---

### Complete Evaluation Pipeline

#### Step 1: Train Master Model

Train on all 42 CapnoBase subjects (for transfer learning pre-training):

```bash
python training/train_master_model.py
```

**Output**: `capnobase_master.pth`

---

#### Step 2: LOSO Cross-Validation

Run 42-fold Leave-One-Subject-Out cross-validation:

```bash
python evaluation/loso_eval.py
```

**Configuration** (from `loso_eval.py`):
- 42 folds (one for each subject)
- Each fold: Train on 41 subjects, test on 1 subject
- Training: Batch size=32, Learning rate=1e-3, 80 epochs per fold
- Inference: Sliding window with stride=30 (1 second) and overlap-add fusion
- **Runtime**: ~2-3 hours on Apple Silicon GPU

**Outputs**:
- `loso_results/fold_00/` through `fold_41/` - Per-fold results
  - `best_model.pth` - Best checkpoint for this fold
  - `training_log.txt` - Training progress
  - `metrics.json` - Final evaluation metrics
  - `loss_curve.png` - Training/validation curves
- `loso_results/all_fold_results.csv` - Aggregated metrics
- `loso_results/aggregated_results.json` - **Median** statistics

**Evaluation Metrics**:
1. **Waveform MAE**: Mean absolute error of normalized signals
2. **Respiratory Rate Error**: FFT-based BPM calculation (6-42 BPM range)
3. **Duty Cycle Error**: Inhalation fraction error (threshold=0.5)

**Resume Support**: Automatically resumes from completed folds. Check progress:
```bash
ls loso_results/fold_*/metrics.json | wc -l
```

---

#### Step 3: PLS Baseline

Run PLS regression baseline with same LOSO splits:

```bash
python evaluation/run_pls_baseline.py
```

**Configuration**:
- PLS components: 26
- Same 42-fold LOSO splits
- Same sliding window inference strategy
- **Runtime**: ~10-15 minutes

**Output**: `pls_results/aggregated_results.json`

---

#### Step 4: Transfer Learning Evaluation

**Zero-Shot Transfer** (no fine-tuning):
```bash
python evaluation/bidmc_zeroshot.py
```

Loads CapnoBase-trained model and directly evaluates on BIDMC (PPG → RESP).

**Output**: `bidmc_results/zeroshot_results.csv`

**Fine-Tuning Transfer**:
```bash
python evaluation/bidmc_finetune.py
```

53-fold LOSO on BIDMC, starting from CapnoBase pre-trained weights.

**Configuration**:
- Learning rate: 1e-4 (10× lower than from-scratch training)
- Epochs: 15 (vs 80 for from-scratch)

**Output**: `bidmc_results/finetune_results.csv`

**Metrics**: Respiratory rate MAE on 30.6s and 60.6s windows

---

#### Step 5: Generate Publication Figures

```bash
# LOSO results visualization
python visualization/generate_paper_plots.py
# Output: paper_plots/ (3-panel figures)

# Deep Learning vs PLS comparison
python visualization/generate_comparison.py
# Output: Comparison plots

# BIDMC transfer learning figures
python visualization/generate_bidmc_figure4a.py
# Output: bidmc_plots/
```

---

## Project Structure

```
Paper_Reproduction/
├── core_model/
│   └── utils.py                    # CorrEncoder architecture (18,441 params)
│
├── preprocessing/
│   ├── preprocess.py               # CapnoBase preprocessing
│   ├── preprocess_bidmc.py         # BIDMC preprocessing
│   └── preprocess_FIR.py           # Alternative FIR filter
│
├── training/
│   ├── train.py                    # Single model (39/3 split)
│   └── train_master_model.py       # All 42 subjects
│
├── evaluation/
│   ├── loso_eval.py                # 42-fold LOSO cross-validation
│   ├── loso_eval_v1_baseline.py    # Alternative baseline evaluation
│   ├── run_pls_baseline.py         # PLS regression baseline
│   ├── bidmc_zeroshot.py           # Zero-shot transfer
│   ├── bidmc_zeroshot_v1.py        # Alternative zero-shot version
│   └── bidmc_finetune.py           # Fine-tuning transfer
│
├── visualization/
│   ├── visualize.py                # Prediction gallery
│   ├── generate_paper_plots.py     # LOSO results figures
│   ├── generate_paper_plots_v1.py  # Alternative plotting version
│   ├── generate_comparison.py      # DL vs PLS comparison
│   └── generate_bidmc_figure4a.py  # BIDMC transfer figures
│
├── utilities/
│   ├── eda.py                      # Exploratory data analysis
│   ├── check_nans.py               # Data validation
│   └── test.py                     # Testing utilities
│
├── Data Directories
│   ├── raw_data/                   # CapnoBase raw CSV files
│   ├── raw_data_bidmc/             # BIDMC raw CSV files
│   ├── processed_data/             # CapnoBase .pt files
│   └── processed_data_bidmc/       # BIDMC .pt files
│
├── Results Directories
│   ├── loso_results/               # LOSO fold results
│   ├── pls_results/                # PLS baseline results
│   ├── bidmc_results/              # Transfer learning results
│   ├── eda_results/                # EDA outputs
│   ├── paper_plots/                # Generated figures
│   └── bidmc_plots/                # BIDMC figures
│
└── Configuration
    ├── pyproject.toml              # Project dependencies
    ├── uv.lock                     # Dependency lock file
    └── README.md                   # This file
```

---

## Model Architecture

**Deep Corr-Encoder** (from `utils.py`):

```
Input: [Batch, 1, 288]  (PPG signals, normalized to [-1, 1])
  ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER                                                     │
├─────────────────────────────────────────────────────────────┤
│ Conv1d:      288 → 179  (kernel=150, pad=20, 1→8 channels) │
│ Dropout:     p=0.5                                          │
│ Activation:  ReLU                                           │
│                                                             │
│ Conv1d:      179 → 125  (kernel=75, pad=10, 8→8 channels)  │
│ Dropout:     p=0.5                                          │
│ Activation:  ReLU                                           │
│                                                             │
│ Conv1d:      125 → 76   (kernel=50, pad=0, 8→8 channels)   │
│ Dropout:     p=0.5                                          │
│ Activation:  Sigmoid                                        │
└─────────────────────────────────────────────────────────────┘
  ↓
Bottleneck: [Batch, 8, 76]
  ↓
┌─────────────────────────────────────────────────────────────┐
│ DECODER                                                     │
├─────────────────────────────────────────────────────────────┤
│ ConvTranspose1d:  76 → 125  (kernel=50, pad=0, 8→8 ch)     │
│ Activation:       Sigmoid                                   │
│                                                             │
│ ConvTranspose1d:  125 → 179 (kernel=75, pad=10, 8→8 ch)    │
│ Activation:       ReLU                                      │
│                                                             │
│ ConvTranspose1d:  179 → 288 (kernel=150, pad=20, 8→1 ch)   │
│ Activation:       ReLU                                      │
└─────────────────────────────────────────────────────────────┘
  ↓
Output: [Batch, 1, 288]  (Respiratory signals, normalized to [0, 1])
```

**Specifications**:
- **Total Parameters**: 18,441 (verified in `utils.py`)
- **Window Duration**: 9.6 seconds @ 30 Hz = 288 samples
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
  - Training from scratch: LR=1e-3
  - Fine-tuning: LR=1e-4
- **Regularization**: Dropout (p=0.5) in encoder only

---

## Evaluation Metrics

### 1. Waveform MAE

Mean absolute error between predicted and ground truth signals:

```
MAE = mean(|prediction - ground_truth|)
```

- Calculated on normalized signals [0, 1]
- Applied to continuous signals after overlap-add fusion

### 2. Respiratory Rate Error (BPM)

FFT-based respiratory rate extraction (from `loso_eval.py`):

**Method**:
1. Remove DC component (subtract mean)
2. Compute FFT and magnitude spectrum
3. Restrict to physiological range: 0.1-0.7 Hz (6-42 BPM)
4. Find peak frequency in magnitude spectrum
5. Convert to BPM: `peak_freq × 60`

**Error Calculation**:
```
RR_error = |RR_predicted - RR_true|
```

### 3. Duty Cycle Error

Inhalation fraction of respiratory cycle:

**Method**:
1. Binarize signal at threshold=0.5 (midpoint of [0,1] normalization)
   - CO2 < 0.5 → Inhalation (binary=1)
   - CO2 ≥ 0.5 → Exhalation (binary=0)
2. Calculate duty cycle: `mean(binary_signal)`

**Error Calculation**:
```
DC_error = |DC_predicted - DC_true|
```

### Evaluation Strategy

- **Sliding Window Inference**: stride=30 samples (1 second at 30 Hz)
- **Overlap-Add Fusion**: Average predictions in overlapping regions (90% overlap)
- **LOSO Aggregation**: **Median** across 42 folds (robust to outliers)

---

## Results Summary

Results from experimental evaluation on CapnoBase and BIDMC datasets:

| Method                       | Waveform MAE | RR Error (BPM) | Duty Cycle Error |
|------------------------------|--------------|----------------|------------------|
| Deep Corr-Encoder (LOSO)     | 0.293        | 0.00           | 0.097            |
| PLS Baseline (n=26)          | 0.355        | 0.00           | 0.286            |
| BIDMC Zero-Shot (30.6s)      | N/A          | 1.96           | N/A              |
| BIDMC Zero-Shot (60.6s)      | N/A          | 1.27           | N/A              |
| BIDMC Fine-Tuned (30.6s)     | N/A          | 0.78           | N/A              |
| BIDMC Fine-Tuned (60.6s)     | N/A          | 0.42           | N/A              |

*Note: All values are median across folds/subjects. Detailed per-fold results available in `loso_results/all_fold_results.csv`, `pls_results/pls_all_fold_results.csv`, and `bidmc_results/*.csv`.*

---

## Troubleshooting

### Issue 1: Device Errors (MPS/CUDA Out of Memory)

**Error**:
```
RuntimeError: MPS backend out of memory
```

**Solution**:
- Reduce `BATCH_SIZE` in `train.py` from 32 to 16
- Device detection is automatic (priority: MPS > CUDA > CPU)

---

### Issue 2: Missing Raw Data Files

**Error**:
```
FileNotFoundError: raw_data/0009_8min_signal.csv
```

**Solution**:
1. Ensure CapnoBase dataset is downloaded from [DOI:10.5683/SP2/NLB8IT](https://doi.org/10.5683/SP2/NLB8IT)
2. Extract all files to `raw_data/` directory
3. Verify you have 42 subjects (84 files: both `*_signal.csv` and `*_param.csv`)

---

### Issue 3: NaN Losses During Training

**Symptom**: Loss becomes NaN after several epochs

**Solution**:
1. Check for NaN values in preprocessed data:
   ```bash
   python check_nans.py
   ```
2. Verify data preprocessing completed successfully
3. Try reducing learning rate

---

### Issue 4: LOSO Evaluation Too Slow

**Symptom**: Each fold takes >10 minutes on CPU

**Solutions**:
1. **Use GPU** if available (MPS/CUDA automatically detected)
2. **Reduce epochs** for testing: Edit `NUM_EPOCHS` in `loso_eval.py`
3. **Test single fold** first: Modify the fold loop to run only fold 0

---

### Issue 5: Preprocessing Fails

**Error**:
```
AssertionError: Expected 42 subjects, found X
```

**Solution**:
1. Verify all subject files are present in `raw_data/`
2. Check that both `*_signal.csv` and `*_param.csv` exist for each subject
3. Ensure no corrupted or partially downloaded files

---

### Issue 6: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
1. Ensure virtual environment is activated:
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```
2. Reinstall dependencies:
   ```bash
   pip install -e .
   ```

---

## Advanced Usage

### Resuming LOSO Evaluation

LOSO automatically resumes from completed folds:

```bash
# Check progress
ls loso_results/fold_*/metrics.json | wc -l

# View specific fold results
cat loso_results/fold_00/metrics.json

# Start fresh (delete all results)
rm -rf loso_results/
```

---

### Custom Hyperparameters

Edit configuration in `train.py`:

```python
BATCH_SIZE = 32          # Try 16, 64
LEARNING_RATE = 1e-3     # Try 5e-4, 1e-4
NUM_EPOCHS = 80          # Try 50, 100
```

Edit LOSO configuration in `loso_eval.py`:

```python
STRIDE = 30              # Sliding window stride (try 15, 60)
NUM_EPOCHS = 80          # Training epochs per fold
```

---

### Visualizing Predictions

Generate prediction gallery with sample predictions:

```bash
python visualization/visualize.py
```

**Output**: `prediction_gallery.png`

---

## Citation

If you use this code or reproduce this work, please cite:

```bibtex
@article{deep_corr_encoder,
  title={Deep Corr-Encoder: [Full Title]},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]},
  doi={[DOI]}
}
```

**Dataset Citations**:

CapnoBase:
```bibtex
@data{capnobase,
  author = {CapnoBase Contributors},
  title = {CapnoBase Dataset},
  year = {2023},
  doi = {10.5683/SP2/NLB8IT},
  url = {https://doi.org/10.5683/SP2/NLB8IT}
}
```

BIDMC: Obtain from official dataset source

---

## License

This reproduction is provided for **academic and research purposes**.

**Dataset Licenses**:
- CapnoBase: See [DOI:10.5683/SP2/NLB8IT](https://doi.org/10.5683/SP2/NLB8IT) for license terms
- BIDMC: See official source for license terms

**Code License**: MIT

---

## Acknowledgments

- Original Deep Corr-Encoder paper authors
- CapnoBase dataset contributors
- BIDMC dataset contributors
- PyTorch, scikit-learn, and scipy communities

---

## Contributing

**Author**: Justin Wang

**Pull Requests Welcome** for:
- Bug fixes
- Documentation improvements
- Additional evaluation metrics
- Visualization enhancements
- Dataset support (new physiological datasets)

**Development Guidelines**:
1. Follow existing code style
2. Add docstrings to new functions
3. Test on CPU, CUDA, and MPS devices
4. Update README for new features

---

**Last Updated**: January 2026
