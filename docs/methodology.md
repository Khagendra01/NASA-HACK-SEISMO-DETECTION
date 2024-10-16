# Planetary Seismic Event Detection - Methodology

## Overview

This document describes the scientific methodology behind the planetary seismic event detection framework.

## Signal Processing Pipeline

### 1. Band-Pass Filtering

We use a 4th-order Butterworth band-pass filter to remove noise outside the seismically meaningful frequency range:

- Lunar data: 0.1 - 5.0 Hz
- Martian data: 0.1 - 10.0 Hz

The filter is applied using zero-phase forward-backward filtering to eliminate phase distortions.

### 2. Empirical Mode Decomposition (EMD)

EMD decomposes the signal into Intrinsic Mode Functions (IMFs) that capture oscillatory behavior at different time scales:

x(t) = Σ IMF_i(t) + r(t)

Each IMF satisfies two conditions:
1. The number of extrema and zero-crossings differ by at most one
2. The envelope defined by local maxima and minima is symmetric

### 3. Energy Extraction

Energy is computed from sliding windows of each IMF:

E(t) = Σ IMF_i(t)²

This emphasizes transient seismic arrivals while suppressing background noise.

### 4. Normalization

Temporal normalization ensures consistent amplitude scaling across different recordings.

## Deep Learning Model

### Architecture

The 1D U-Net architecture consists of:

- **Encoder**: 3 encoder blocks with increasing filter counts (32, 64, 128)
- **Bottleneck**: Residual block with SE attention
- **Decoder**: 3 decoder blocks with skip connections
- **Output**: Sigmoid activation for probability prediction

### Key Components

1. **Residual Blocks**: Enable gradient flow through deep networks
2. **SE Attention**: Channel-wise recalibration of feature maps
3. **Skip Connections**: Preserve spatial resolution during upsampling

### Training

- Loss: Mean Squared Error (MSE) between predicted and Gaussian target
- Optimizer: Adam with learning rate 1e-3
- Early stopping based on validation loss

## Explainable AI

### Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) generates temporal attention maps showing which waveform regions contributed most strongly to the detection decision.

The method computes:
1. Gradient of target output with respect to final convolutional layer
2. Channel-wise weighting of activation maps
3. ReLU-activated weighted combination

### Interpretation

The resulting heatmaps enable:
- Event localization verification
- Model debugging
- Scientific interpretability
- IMF contribution analysis

## Metrics

### Primary Metrics

1. **Precision**: Fraction of correct detections among all positive predictions
2. **Recall**: Fraction of true events detected
3. **F1 Score**: Harmonic mean of precision and recall
4. **IoU**: Intersection over Union for temporal localization

### Secondary Metrics

1. **Peak Localization Error**: Temporal distance between predicted and true event peaks
2. **Mean Absolute Error**: Overall waveform-level prediction accuracy

## References

1. Huang et al. (1998). The Empirical Mode Decomposition and the Hilbert Spectrum.
2. Selvaraj et al. (2023). Deep Learning for Seismic Event Detection.
3. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.