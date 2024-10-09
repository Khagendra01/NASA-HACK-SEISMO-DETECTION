import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import obspy
import scipy.signal as sg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import UNet1D
from train import train, set_seed, DEVICE
from preprocessing import preprocess_signal, butter_bandpass_filter, create_gaussian_labels


def load_catalog(catalog_path):
    return pd.read_csv(catalog_path)


def process_file(args):
    file_path, arrival_time, lowcut, highcut, window_size_sec, resample_size = args

    if not os.path.exists(file_path):
        return None, None

    stream = obspy.read(file_path)
    data = stream[0].data
    fs = stream[0].stats.sampling_rate

    imfs = preprocess_signal(data, fs, lowcut, highcut, window_size_sec, resample_size=resample_size)

    label = create_gaussian_labels(len(data), int(arrival_time * fs), sigma=100)
    label = sg.resample(label, resample_size)

    return imfs, label


def prepare_moon_data(catalog_path, data_dir, lowcut=0.4, highcut=1.2, window_size_sec=1800):
    catalog = load_catalog(catalog_path)

    file_paths = []
    labels = []

    for idx, row in catalog.iterrows():
        filename = row['filename'] + '.mseed'
        file_path = os.path.join(data_dir, filename)
        arrival_time = row['time_rel(sec)']
        file_paths.append((file_path, arrival_time, lowcut, highcut, window_size_sec, 4000))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    valid_results = [(r[0], r[1]) for r in results if r[0] is not None]

    if not valid_results:
        raise ValueError("No valid data found")

    X = np.array([r[0] for r in valid_results])
    Y = np.array([r[1] for r in valid_results])

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)

    return X, Y


def main():
    set_seed(42)

    catalog_path = '../data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
    data_dir = '../data/lunar/training/data/S12_GradeA'

    print("Loading and preprocessing data...")
    X, Y = prepare_moon_data(catalog_path, data_dir)

    print(f"Data shape: X={X.shape}, Y={Y.shape}")

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = UNet1D(in_channels=5).to(DEVICE)

    checkpoint_dir = '../outputs/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=0.001,
        checkpoint_path=checkpoint_path
    )

    print(f"Training complete. Model saved to {checkpoint_path}")


if __name__ == '__main__':
    main()
