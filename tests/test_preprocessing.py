import pytest
import numpy as np
from src.preprocessing import (
    butter_bandpass_filter,
    emd_decompose,
    compute_imf_energy,
    normalize_signal,
    gaussian_target
)


def test_butter_bandpass_filter():
    fs = 100
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

    filtered = butter_bandpass_filter(signal, 0.5, 2.5, fs)

    assert filtered.shape == signal.shape
    assert not np.any(np.isnan(filtered))


def test_gaussian_target():
    length = 100
    center = 50
    sigma = 10

    target = gaussian_target(length, center, sigma)

    assert target.shape == (length,)
    assert target.max() == pytest.approx(1.0)
    assert target[center] == pytest.approx(1.0)


def test_normalize_signal():
    signal = np.random.randn(100)

    normalized = normalize_signal(signal)

    assert normalized.shape == signal.shape
    assert np.abs(normalized.mean()) < 0.1
    assert normalized.std() == pytest.approx(1.0, rel=0.01)


def test_emd_decompose():
    fs = 100
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 7 * t)

    imfs = emd_decompose(signal, max_imfs=5)

    assert imfs.shape[0] == len(signal)
    assert imfs.shape[1] >= 2


def test_compute_imf_energy():
    imfs = np.random.randn(100, 4)
    window_size = 10

    energy = compute_imf_energy(imfs, window_size)

    assert energy.shape == imfs.shape
    assert np.all(energy >= 0)