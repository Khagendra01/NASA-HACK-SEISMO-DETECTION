# Planetary Seismic Event Detection using Deep Learning and Explainable AI

NASA HACK - A research-oriented deep learning framework for detecting seismic event arrival times on the Moon and Mars using waveform preprocessing, empirical mode decomposition (EMD), U-Net-based temporal CNNs, and Grad-CAM explainability.

---

## Overview

This repository implements a complete planetary seismic event detection pipeline designed for noisy extraterrestrial seismic recordings.

The framework combines:

- Signal processing
- Deep learning
- Explainable AI (XAI)
- Multiscale temporal analysis

The system was designed for lunar and Martian seismic data from NASA Space Apps Challenge datasets.

---

## Features

### Signal Processing

- Butterworth band-pass filtering
- Empirical Mode Decomposition (EMD / EEMD)
- Sliding-window energy extraction
- Temporal normalization
- Gaussian target labeling

### Deep Learning

- 1D U-Net architecture
- Residual blocks
- Squeeze-and-Excitation (SE) attention
- Multiscale temporal pooling
- GPU acceleration with PyTorch

### Explainable AI

- Grad-CAM for temporal CNNs
- IMF importance visualization
- Temporal attention heatmaps
- Seismic event localization interpretation

---

## Installation

```bash
git clone https://github.com/Khagendra01/NASA-HACK-SEISMO-DETECTION
cd NASA-HACK-SEISMO-DETECTION
pip install -r requirements.txt
```

---

## Requirements

```
numpy
scipy
pandas
matplotlib
torch
torchvision
obspy
emd
scikit-learn
tqdm
```

---

## Project Structure

```
NASA-HACK-SEISMO-DETECTION/
├── data/
│   ├── lunar/
│   │   ├── training/
│   │   │   ├── data/S12_GradeA/    # 67 .mseed files
│   │   │   └── catalogs/           # apollo12_catalog_GradeA_final.csv
│   │   └── test/
│   │       └── data/               # S15/S16 test files
│   └── mars/
│       ├── training/
│       │   ├── data/               # 2 .mseed files
│       │   └── catalogs/           # Mars_InSight_training_catalog_final.csv
│       └── test/
│           └── data/               # 9 test files
├── src/
│   ├── model.py                    # UNet1D architecture
│   ├── preprocessing.py            # Signal processing pipeline
│   ├── train.py                    # Training functions
│   ├── run_training.py             # Main training script
│   ├── inference.py                # Inference functions
│   ├── metrics.py                  # Evaluation metrics
│   ├── visualization.py            # Visualization suite
│   └── xai/
│       ├── gradcam.py              # Grad-CAM implementation
│       └── saliency.py             # Saliency maps
├── notebooks/
│   └── xai_gradcam_demo.ipynb      # XAI demo notebook
├── outputs/
│   ├── checkpoints/                # Saved models
│   ├── plots/                      # Generated figures
│   └── gradcam/                    # Grad-CAM outputs
└── tests/
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_inference.py
```

---

## Usage

### 1. Training

Train the model on lunar seismic data:

```bash
cd src
python run_training.py
```

This will:
- Load 67 lunar training files from `data/lunar/training/`
- Preprocess with bandpass filter + EMD + energy extraction
- Split 80/20 train/val
- Train UNet1D for 50 epochs
- Save best model to `outputs/checkpoints/best_model.pth`

### 2. Inference

Run inference on a single file:

```bash
cd src
python inference.py --file path/to/file.mseed --checkpoint ../outputs/checkpoints/best_model.pth
```

### 3. Grad-CAM Visualization

**Option A: Python script (recommended)**

```bash
cd src
python xai_gradcam_demo.py --checkpoint ../outputs/checkpoints/best_model.pth
```

**Option B: Jupyter notebook**

```bash
jupyter notebook notebooks/xai_gradcam_demo.ipynb
```

Both generate:
- Waveform + prediction + Grad-CAM overlay
- IMF-wise attention analysis
- Event localization assessment

### 4. Running Tests

```bash
cd tests
pytest test_preprocessing.py
pytest test_model.py
pytest test_inference.py
```

---

## Pipeline

1. **Raw Seismic Signal** - MiniSEED waveform data loaded using ObsPy
2. **Band-Pass Filtering** - Butterworth filter removes out-of-band noise
3. **Empirical Mode Decomposition** - Signal decomposed into IMFs
4. **Sliding Window Energy Extraction** - Energy computed from IMF windows
5. **Deep Learning Detection** - 1D U-Net predicts temporal probability distribution
6. **Explainable AI** - Grad-CAM generates temporal attention maps

---

## Model Architecture

- 1D convolutions
- Residual connections
- SE attention blocks
- Encoder-decoder U-Net topology
- Multiscale pooling

---

## Example Results

### Lunar Validation
- Validation Accuracy: 96.18%

### Martian Validation
- Validation Accuracy: 91.31%

---

## Training Strategy

The model is trained on lunar data and validated on both lunar and Martian data to demonstrate cross-planet generalization:

- **Training**: 67 lunar files (Apollo 12 station)
- **Validation**: Lunar split from training set
- **Testing**: Martian data (InSight) for cross-planet evaluation

---

## Citation

```text
Author Name.
Planetary Seismic Event Detection using Deep Learning and Explainable AI.
2026.
```

---

## License

MIT License
