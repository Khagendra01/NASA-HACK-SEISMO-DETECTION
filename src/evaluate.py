import os
import sys
import numpy as np
import torch
import obspy
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import UNet1D
from preprocessing import preprocess_signal
from metrics import localization_error, precision_recall


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_dataset(model, data_dir, catalog_path=None, lowcut=0.4, highcut=1.2):
    results = []

    if catalog_path and os.path.exists(catalog_path):
        catalog = pd.read_csv(catalog_path)
        for idx, row in catalog.iterrows():
            filename = row['filename'] + '.mseed'
            file_path = os.path.join(data_dir, filename)
            arrival_time = row['time_rel(sec)']
            if os.path.exists(file_path):
                results.append((file_path, arrival_time))
    else:
        for root, dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.mseed'):
                    file_path = os.path.join(root, f)
                    csv_path = file_path.replace('.mseed', '.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if 'time_rel(sec)' in df.columns:
                            results.append((file_path, float(df['time_rel(sec)'].iloc[0])))

    loc_errors = []
    all_preds = []
    all_labels = []

    for file_path, arrival_time in tqdm(results, desc='Evaluating'):
        try:
            stream = obspy.read(file_path)
            data = stream[0].data
            fs = stream[0].stats.sampling_rate

            processed = preprocess_signal(data, fs, lowcut, highcut, 1800, resample_size=4000)
            x = torch.tensor(processed[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                pred = model(x).squeeze().cpu().numpy().flatten()

            pred_time = np.argmax(pred) * len(data) / fs / len(pred)
            error = abs(pred_time - arrival_time)
            loc_errors.append(error)

            threshold = 0.25
            pred_binary = (pred > threshold).astype(int)
            label = np.zeros(len(pred))
            arrival_idx = int(arrival_time / (len(data) / fs) * len(pred))
            if arrival_idx < len(label):
                window = 50
                start = max(0, arrival_idx - window)
                end = min(len(label), arrival_idx + window)
                label[start:end] = 1
            all_preds.append(pred_binary)
            all_labels.append(label)

        except Exception as e:
            print(f'Error: {e}')
            continue

    if not loc_errors:
        return None

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    precision, recall = precision_recall(all_preds, all_labels)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'mean_loc_error': np.mean(loc_errors),
        'median_loc_error': np.median(loc_errors),
        'std_loc_error': np.std(loc_errors),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_files': len(loc_errors)
    }


def main():
    checkpoint_path = '../outputs/checkpoints/best_model.pth'

    model = UNet1D(in_channels=5).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print('=== Lunar Test Set ===')
    lunar_metrics = evaluate_dataset(
        model,
        '../data/lunar/test/data',
        lowcut=0.4, highcut=1.2
    )

    if lunar_metrics:
        print(f'Files evaluated: {lunar_metrics["n_files"]}')
        print(f'Mean localization error: {lunar_metrics["mean_loc_error"]:.2f}s')
        print(f'Median localization error: {lunar_metrics["median_loc_error"]:.2f}s')
        print(f'Precision: {lunar_metrics["precision"]:.4f}')
        print(f'Recall: {lunar_metrics["recall"]:.4f}')
        print(f'F1: {lunar_metrics["f1"]:.4f}')

    print('\n=== Mars Test Set ===')
    mars_metrics = evaluate_dataset(
        model,
        '../data/mars/test/data',
        lowcut=2.0, highcut=8.0
    )

    if mars_metrics:
        print(f'Files evaluated: {mars_metrics["n_files"]}')
        print(f'Mean localization error: {mars_metrics["mean_loc_error"]:.2f}s')
        print(f'Median localization error: {mars_metrics["median_loc_error"]:.2f}s')
        print(f'Precision: {mars_metrics["precision"]:.4f}')
        print(f'Recall: {mars_metrics["recall"]:.4f}')
        print(f'F1: {mars_metrics["f1"]:.4f}')

    print('\n=== Summary ===')
    if lunar_metrics:
        print(f'Lunar F1: {lunar_metrics["f1"]:.4f} | Loc Error: {lunar_metrics["mean_loc_error"]:.2f}s')
    if mars_metrics:
        print(f'Mars F1: {mars_metrics["f1"]:.4f} | Loc Error: {mars_metrics["mean_loc_error"]:.2f}s')


if __name__ == '__main__':
    main()
