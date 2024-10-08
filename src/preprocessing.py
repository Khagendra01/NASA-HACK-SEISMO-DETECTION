import numpy as np
import scipy.signal as sg
import emd


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = sg.butter(order, [low, high], btype='band')

    return sg.filtfilt(b, a, data)



def butter_bandstop_filter(data, lowcut, highcut, fs, order=6):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = sg.butter(order, [low, high], btype='bandstop')

    return sg.filtfilt(b, a, data)



def apply_emd(data, max_imfs=5):

    return emd.sift.ensemble_sift(data, max_imfs=max_imfs)



def compute_window_energy(imfs, window_size, stride):

    energy = np.lib.stride_tricks.sliding_window_view(
        imfs ** 2,
        window_shape=window_size,
        axis=0
    )[::stride, :, :]

    energy = np.sum(energy, axis=-1)

    return energy



def normalize_energy(energy):

    max_values = np.max(energy, axis=0)

    max_values[max_values == 0] = 1e-8

    return energy / max_values



def preprocess_signal(
    data,
    fs,
    lowcut,
    highcut,
    window_size_sec,
    resample_size=4000,
    max_imfs=5
):

    filtered = butter_bandpass_filter(
        data,
        lowcut,
        highcut,
        fs
    )

    imfs = apply_emd(filtered, max_imfs=max_imfs)

    window_size = int(window_size_sec * fs)

    stride = window_size // 48

    energy = compute_window_energy(
        imfs,
        window_size,
        stride
    )

    energy = sg.resample(energy, resample_size, axis=0)

    while energy.shape[1] < max_imfs:

        energy = np.concatenate([
            energy,
            np.zeros((energy.shape[0], 1)) + 0.001
        ], axis=1)

    energy = normalize_energy(energy)

    return energy.T



def create_gaussian_labels(
    signal_length,
    arrival_idx,
    sigma
):

    output = np.zeros(signal_length)

    gaussian_window_size = int(sigma * 6)

    if gaussian_window_size % 2 == 1:
        gaussian_window_size += 1

    gaussian_window = sg.windows.gaussian(
        gaussian_window_size,
        std=sigma
    )

    left_idx = arrival_idx - gaussian_window_size // 2
    right_idx = arrival_idx + gaussian_window_size // 2

    left_idx = max(left_idx, 0)
    right_idx = min(right_idx, signal_length)

    output[left_idx:right_idx] = gaussian_window[:right_idx-left_idx]

    return output