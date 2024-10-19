import os
import sys
import numpy as np
import torch
import obspy
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import UNet1D
from preprocessing import preprocess_signal
from xai.gradcam import SeismicGradCAM
from visualization import SeismicVisualizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    model = UNet1D(in_channels=5).to(DEVICE)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
    return model


def process_file(file_path, lowcut, highcut):
    stream = obspy.read(file_path)
    data = stream[0].data
    fs = stream[0].stats.sampling_rate
    processed = preprocess_signal(data, fs, lowcut, highcut, 1800, resample_size=4000)
    return data, fs, processed


def generate_gradcam(model, processed):
    x = torch.tensor(processed[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy().flatten()
    target_layer = model.encoder3.se
    gradcam_gen = SeismicGradCAM(model, target_layer)
    cam = gradcam_gen.compute(x, original_length=len(pred))
    return pred, cam


def generate_all_outputs():
    checkpoint_path = '../outputs/checkpoints/best_model.pth'
    model = load_model(checkpoint_path)

    output_dirs = [
        '../outputs/plots',
        '../outputs/gradcam',
        '../outputs/predictions/lunar',
        '../outputs/predictions/mars'
    ]
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    viz = SeismicVisualizer(save_dir='../outputs/plots')

    print('Generating Grad-CAM outputs for lunar test files...')
    lunar_dir = '../data/lunar/test/data'
    lunar_files = []
    for root, dirs, files in os.walk(lunar_dir):
        for f in files:
            if f.endswith('.mseed'):
                lunar_files.append(os.path.join(root, f))
    lunar_files = sorted(lunar_files)[:10]

    for i, file_path in enumerate(lunar_files):
        try:
            data, fs, processed = process_file(file_path, 0.4, 1.2)
            pred, cam = generate_gradcam(model, processed)
            t = np.linspace(0, len(data) / fs, len(pred))

            viz.plot_waveform_prediction_gradcam(
                time=t, waveform=data.flatten()[:len(t)], prediction=pred, gradcam=cam,
                title=f'Lunar: {Path(file_path).stem[:30]}',
                save_path=f'../outputs/gradcam/lunar_{i:03d}.png'
            )
            print(f'  Generated: lunar_{i:03d}.png')
        except Exception as e:
            print(f'  Error: {Path(file_path).name}: {e}')

    print('Generating Grad-CAM outputs for mars test files...')
    mars_dir = '../data/mars/test/data'
    mars_files = []
    for root, dirs, files in os.walk(mars_dir):
        for f in files:
            if f.endswith('.mseed'):
                mars_files.append(os.path.join(root, f))
    mars_files = sorted(mars_files)

    for i, file_path in enumerate(mars_files):
        try:
            data, fs, processed = process_file(file_path, 2.0, 8.0)
            pred, cam = generate_gradcam(model, processed)
            t = np.linspace(0, len(data) / fs, len(pred))

            viz.plot_waveform_prediction_gradcam(
                time=t, waveform=data.flatten()[:len(t)], prediction=pred, gradcam=cam,
                title=f'Mars: {Path(file_path).stem[:30]}',
                save_path=f'../outputs/gradcam/mars_{i:03d}.png'
            )
            print(f'  Generated: mars_{i:03d}.png')
        except Exception as e:
            print(f'  Error: {Path(file_path).name}: {e}')

    print('Generating comparison plots...')
    generate_comparison_plot(model)

    print('Generating IMF attention plots...')
    generate_imf_attention(model)

    print('All outputs generated in outputs/ folder')


def generate_comparison_plot(model):
    lunar_dir = '../data/lunar/test/data'
    mars_dir = '../data/mars/test/data'

    lunar_files = []
    for root, dirs, files in os.walk(lunar_dir):
        for f in files:
            if f.endswith('.mseed'):
                lunar_files.append(os.path.join(root, f))

    mars_files = []
    for root, dirs, files in os.walk(mars_dir):
        for f in files:
            if f.endswith('.mseed'):
                mars_files.append(os.path.join(root, f))

    lunar_file = sorted(lunar_files)[0]
    mars_file = sorted(mars_files)[0]

    lunar_data, lunar_fs, lunar_processed = process_file(lunar_file, 0.4, 1.2)
    mars_data, mars_fs, mars_processed = process_file(mars_file, 2.0, 8.0)

    lunar_pred, lunar_cam = generate_gradcam(model, lunar_processed)
    mars_pred, mars_cam = generate_gradcam(model, mars_processed)

    lunar_t = np.linspace(0, len(lunar_data) / lunar_fs, len(lunar_pred))
    mars_t = np.linspace(0, len(mars_data) / mars_fs, len(mars_pred))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(lunar_t, lunar_data.flatten()[:len(lunar_t)], color='#3498db', linewidth=0.5)
    axes[0, 0].set_title('Lunar Waveform', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')

    axes[0, 1].plot(lunar_t, lunar_pred, color='#e74c3c', linewidth=1.5)
    axes[0, 1].fill_between(lunar_t, 0, lunar_pred, alpha=0.3, color='#e74c3c')
    axes[0, 1].set_title('Lunar Prediction', fontweight='bold')
    axes[0, 1].set_ylim(0, 1)

    im0 = axes[0, 2].imshow(lunar_cam.reshape(1, -1), aspect='auto', cmap='hot',
                             extent=[lunar_t[0], lunar_t[-1], 0, 1])
    axes[0, 2].set_title('Lunar Grad-CAM Heatmap', fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 2])

    axes[1, 0].plot(mars_t, mars_data.flatten()[:len(mars_t)], color='#e67e22', linewidth=0.5)
    axes[1, 0].set_title('Mars Waveform', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')

    axes[1, 1].plot(mars_t, mars_pred, color='#e74c3c', linewidth=1.5)
    axes[1, 1].fill_between(mars_t, 0, mars_pred, alpha=0.3, color='#e74c3c')
    axes[1, 1].set_title('Mars Prediction', fontweight='bold')
    axes[1, 1].set_ylim(0, 1)

    im1 = axes[1, 2].imshow(mars_cam.reshape(1, -1), aspect='auto', cmap='hot',
                             extent=[mars_t[0], mars_t[-1], 0, 1])
    axes[1, 2].set_title('Mars Grad-CAM Heatmap', fontweight='bold')
    plt.colorbar(im1, ax=axes[1, 2])

    plt.suptitle('Cross-Planet Comparison: Lunar vs Martian Seismic Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/plots/comparison_moon_mars.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  Generated: comparison_moon_mars.png')


def generate_imf_attention(model):
    lunar_dir = '../data/lunar/test/data'
    lunar_files = []
    for root, dirs, files in os.walk(lunar_dir):
        for f in files:
            if f.endswith('.mseed'):
                lunar_files.append(os.path.join(root, f))
    lunar_file = sorted(lunar_files)[0]

    data, fs, processed = process_file(lunar_file, 0.4, 1.2)
    pred, cam = generate_gradcam(model, processed)

    t = np.linspace(0, len(data) / fs, len(pred))

    viz = SeismicVisualizer(save_dir='../outputs/plots')
    viz.plot_imf_attention(
        time=t, imf_energy=processed, gradcam=cam,
        title='IMF-wise Attention Analysis',
        save_path='../outputs/plots/imf_attention.png'
    )
    print('  Generated: imf_attention.png')


if __name__ == '__main__':
    generate_all_outputs()
