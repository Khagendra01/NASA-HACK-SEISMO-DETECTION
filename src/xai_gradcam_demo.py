import sys
import os
import numpy as np
import torch
import scipy.signal as sg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import UNet1D
from xai.gradcam import GradCAM1D, SeismicGradCAM
from visualization import SeismicVisualizer
from preprocessing import preprocess_signal


def create_synthetic_signal(fs=100, duration=10, arrival_time=5.0):
    t = np.linspace(0, duration, int(fs * duration))
    noise = 0.1 * np.random.randn(len(t))
    arrival = np.exp(-0.5 * ((t - arrival_time) / 0.3)**2) * np.sin(2 * np.pi * 5 * (t - arrival_time))
    waveform = noise + arrival
    arrival_idx = int(arrival_time * fs)
    return t, waveform, arrival_idx


def run_gradcam_demo(checkpoint_path=None, save_dir='outputs/plots'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet1D(in_channels=5)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded model from {checkpoint_path}')
    else:
        print('Using randomly initialized model (demo only)')

    model = model.to(device)
    model.eval()

    t, waveform, arrival_idx = create_synthetic_signal()

    processed = preprocess_signal(waveform, 100, 0.5, 20.0, 1.0)
    x = torch.tensor(processed[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(-1)

    pred_interp = np.interp(t, np.linspace(t[0], t[-1], len(pred)), pred)

    target_layer = model.encoder3.se
    gradcam_gen = SeismicGradCAM(model, target_layer)
    cam = gradcam_gen.compute(x, original_length=len(t))

    viz = SeismicVisualizer(save_dir=save_dir)

    viz.plot_waveform_prediction_gradcam(
        time=t, waveform=waveform, prediction=pred_interp, gradcam=cam,
        true_arrival_idx=arrival_idx, title='Grad-CAM Seismic Event Detection',
        save_path=os.path.join(save_dir, 'gradcam_demo.png')
    )

    viz.plot_imf_attention(
        time=t, imf_energy=processed, gradcam=cam,
        title='IMF Contribution to Detection',
        save_path=os.path.join(save_dir, 'gradcam_imf.png')
    )

    pred_arrival_idx = np.argmax(pred_interp)
    viz.plot_arrival_localization(
        time=t, waveform=waveform, prediction=pred_interp,
        true_arrival_idx=arrival_idx, pred_arrival_idx=pred_arrival_idx,
        confidence=pred_interp[pred_arrival_idx],
        title='Event Localization',
        save_path=os.path.join(save_dir, 'gradcam_localization.png')
    )

    print(f'\nDemo complete. Figures saved to {save_dir}/')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Grad-CAM Demo for Seismic Detection')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--save-dir', type=str, default='outputs/plots', help='Output directory')
    args = parser.parse_args()
    run_gradcam_demo(checkpoint_path=args.checkpoint, save_dir=args.save_dir)
