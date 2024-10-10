import os
import sys
import numpy as np
import torch
import obspy
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import UNet1D
from preprocessing import preprocess_signal
from xai.gradcam import SeismicGradCAM
from visualization import SeismicVisualizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_test_files(data_dir):
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.mseed'):
                files.append(os.path.join(root, f))
    return sorted(files)


def load_true_arrival(file_path):
    csv_path = file_path.replace('.mseed', '.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'time_rel(sec)' in df.columns:
            return float(df['time_rel(sec)'].iloc[0])
    return None


def process_single_file(model, file_path, save_dir, viz):
    try:
        stream = obspy.read(file_path)
        data = stream[0].data
        fs = stream[0].stats.sampling_rate

        processed = preprocess_signal(data, fs, 0.4, 1.2, 1800, resample_size=4000)
        x = torch.tensor(processed[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred = model(x).squeeze().cpu().numpy().flatten()

        t = np.linspace(0, len(data) / fs, len(pred))
        waveform = data.flatten()[:len(t)]

        target_layer = model.encoder3.se
        gradcam_gen = SeismicGradCAM(model, target_layer)
        cam = gradcam_gen.compute(x, original_length=len(t))

        true_arrival = load_true_arrival(file_path)

        base_name = Path(file_path).stem
        sample_dir = os.path.join(save_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)

        viz.plot_waveform_prediction_gradcam(
            time=t, waveform=waveform, prediction=pred, gradcam=cam,
            true_arrival_idx=int(true_arrival * fs) if true_arrival else None,
            title=f'Detection: {base_name}',
            save_path=os.path.join(sample_dir, 'gradcam.png')
        )

        pred_arrival_idx = np.argmax(pred)
        viz.plot_arrival_localization(
            time=t, waveform=waveform, prediction=pred,
            true_arrival_idx=int(true_arrival * fs) if true_arrival else pred_arrival_idx,
            pred_arrival_idx=pred_arrival_idx,
            confidence=pred[pred_arrival_idx],
            title=f'Localization: {base_name}',
            save_path=os.path.join(sample_dir, 'localization.png')
        )

        return True

    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return False


def main():
    checkpoint_path = '../outputs/checkpoints/best_model.pth'

    model = UNet1D(in_channels=5).to(DEVICE)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f'Loaded model from {checkpoint_path}')
    else:
        print('No checkpoint found. Train first.')
        return

    model.eval()

    viz = SeismicVisualizer(save_dir='../outputs/plots')

    lunar_test_dir = '../data/lunar/test/data'
    mars_test_dir = '../data/mars/test/data'

    lunar_files = find_test_files(lunar_test_dir)
    mars_files = find_test_files(mars_test_dir)

    print(f'\nFound {len(lunar_files)} lunar test files')
    print(f'Found {len(mars_files)} mars test files')

    save_dir = '../outputs/predictions'
    os.makedirs(save_dir, exist_ok=True)

    success_count = 0
    total_count = 0

    print('\nProcessing lunar files...')
    for f in tqdm(lunar_files, desc='Lunar'):
        total_count += 1
        if process_single_file(model, f, save_dir, viz):
            success_count += 1

    print('\nProcessing mars files...')
    for f in tqdm(mars_files, desc='Mars'):
        total_count += 1
        if process_single_file(model, f, save_dir, viz):
            success_count += 1

    print(f'\nDone: {success_count}/{total_count} files processed')
    print(f'Results saved to {save_dir}/')


if __name__ == '__main__':
    main()
