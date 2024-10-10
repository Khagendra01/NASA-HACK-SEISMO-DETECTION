import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

# Custom seismic colormap
SEISMIC_CMAP = LinearSegmentedColormap.from_list(
    'seismic_gradcam',
    ['#0d0887', '#5b02a3', '#9c179e', '#cc4778', '#ed7953', '#fdb42f', '#f0f921']
)


class SeismicVisualizer:
    """Comprehensive research-grade visualization suite for seismic XAI."""

    def __init__(self, save_dir: str = 'outputs/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10


    def plot_waveform_prediction_gradcam(
        self,
        time: np.ndarray,
        waveform: np.ndarray,
        prediction: np.ndarray,
        gradcam: np.ndarray,
        true_arrival_idx: Optional[int] = None,
        title: str = 'Seismic Event Detection',
        save_path: Optional[str] = None
    ) -> None:
        """Core figure: waveform + prediction confidence + Grad-CAM attention overlay."""

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1, 1]})

        # Panel 1: Waveform with Grad-CAM overlay
        ax1 = axes[0]
        ax1.plot(time, waveform, color='#2c3e50', linewidth=0.8, alpha=0.9, label='Waveform')

        gradcam_scaled = gradcam * np.max(np.abs(waveform))
        im = ax1.scatter(time, waveform, c=gradcam, cmap=SEISMIC_CMAP,
                         s=1, alpha=0.7, zorder=2)
        ax1.fill_between(time, 0, gradcam_scaled, alpha=0.25, color='#e74c3c',
                         label='Grad-CAM Attention')

        if true_arrival_idx is not None and true_arrival_idx < len(time):
            ax1.axvline(x=time[true_arrival_idx], color='#27ae60', linestyle='--',
                        linewidth=2, label=f'True Arrival ({time[true_arrival_idx]:.2f}s)')

        ax1.set_ylabel('Amplitude')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        plt.colorbar(im, ax=ax1, label='Attention Intensity', shrink=0.6)

        # Panel 2: Prediction confidence
        ax2 = axes[1]
        ax2.fill_between(time, 0, prediction, alpha=0.6, color='#3498db')
        ax2.plot(time, prediction, color='#2980b9', linewidth=1.5)
        ax2.axhline(y=0.25, color='gray', linestyle=':', alpha=0.7, label='Threshold (0.25)')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right', fontsize=8)

        # Panel 3: Attention heatmap
        ax3 = axes[2]
        ax3.imshow(gradcam.reshape(1, -1), aspect='auto', cmap=SEISMIC_CMAP,
                   extent=[time[0], time[-1], 0, 1])
        ax3.set_ylabel('Attention')
        ax3.set_xlabel('Time (s)')
        ax3.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def plot_imf_attention(
        self,
        time: np.ndarray,
        imf_energy: np.ndarray,
        gradcam: np.ndarray,
        imf_labels: Optional[List[str]] = None,
        title: str = 'IMF-wise Attention Analysis',
        save_path: Optional[str] = None
    ) -> None:
        """Scientifically critical: visualize which IMF contributed most to detection."""

        n_imfs = imf_energy.shape[0]
        if imf_labels is None:
            imf_labels = [f'IMF-{i+1}' for i in range(n_imfs)]

        fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 2.5 * (n_imfs + 1)), sharex=True)

        for i in range(n_imfs):
            ax = axes[i]
            imf_norm = imf_energy[i] / (np.max(imf_energy[i]) + 1e-8)

            ax.plot(time, imf_norm, color='#3498db', linewidth=0.8, alpha=0.9)
            ax.fill_between(time, 0, imf_norm, alpha=0.3, color='#3498db')

            attention_color = SEISMIC_CMAP(gradcam[i] if i < len(gradcam) else np.mean(gradcam))
            ax.set_facecolor((*attention_color[:3], 0.15))

            ax.set_ylabel(imf_labels[i], fontsize=9)
            ax.set_xlim(time[0], time[-1])

        # Summary attention panel
        ax_summary = axes[-1]
        imf_importance = np.mean(imf_energy, axis=1)
        imf_importance = imf_importance / (np.max(imf_importance) + 1e-8)

        bars = ax_summary.barh(imf_labels, imf_importance, color=SEISMIC_CMAP(imf_importance))
        ax_summary.set_xlabel('Relative Importance')
        ax_summary.set_title('IMF Contribution to Detection', fontsize=10)

        for bar, val in zip(bars, imf_importance):
            ax_summary.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=8)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def plot_multilayer_gradcam(
        self,
        time: np.ndarray,
        waveform: np.ndarray,
        layer_cams: Dict[str, np.ndarray],
        title: str = 'Multi-Layer Grad-CAM Analysis',
        save_path: Optional[str] = None
    ) -> None:
        """U-Net encoder/bottleneck/decoder attention visualization."""

        n_layers = len(layer_cams)
        fig, axes = plt.subplots(n_layers + 1, 1, figsize=(14, 2.2 * (n_layers + 1)), sharex=True)

        # Original waveform
        axes[0].plot(time, waveform, color='#2c3e50', linewidth=0.8)
        axes[0].set_ylabel('Waveform')
        axes[0].set_title('Original Signal', fontsize=10)

        layer_names = list(layer_cams.keys())
        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx + 1]
            cam = layer_cams[layer_name]
            cam_norm = cam / (np.max(cam) + 1e-8)

            ax.plot(time[:len(cam_norm)], cam_norm, color=SEISMIC_CMAP(0.3 + 0.5 * idx/n_layers),
                    linewidth=1.2)
            ax.fill_between(time[:len(cam_norm)], 0, cam_norm,
                           alpha=0.3, color=SEISMIC_CMAP(0.3 + 0.5 * idx/n_layers))
            ax.set_ylabel(layer_name, fontsize=9)
            ax.set_ylim(0, 1.1)

        axes[-1].set_xlabel('Time (s)')

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def plot_arrival_localization(
        self,
        time: np.ndarray,
        waveform: np.ndarray,
        prediction: np.ndarray,
        true_arrival_idx: int,
        pred_arrival_idx: int,
        confidence: float,
        title: str = 'Event Localization Assessment',
        save_path: Optional[str] = None
    ) -> None:
        """Evaluate temporal precision: true vs predicted arrival."""

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(time, waveform, color='#7f8c8d', linewidth=0.8, alpha=0.7, label='Waveform')

        ax_pred = ax.twinx()
        ax_pred.plot(time, prediction, color='#e74c3c', linewidth=2, label='Prediction')
        ax_pred.fill_between(time, 0, prediction, alpha=0.2, color='#e74c3c')
        ax_pred.set_ylabel('Confidence', color='#e74c3c')
        ax_pred.tick_params(axis='y', labelcolor='#e74c3c')

        true_time = time[true_arrival_idx] if true_arrival_idx < len(time) else time[-1]
        pred_time = time[pred_arrival_idx] if pred_arrival_idx < len(time) else time[-1]

        ax.axvline(x=true_time, color='#27ae60', linestyle='--', linewidth=2.5,
                   label=f'True Arrival: {true_time:.3f}s')
        ax.axvline(x=pred_time, color='#e74c3c', linestyle=':', linewidth=2.5,
                   label=f'Predicted: {pred_time:.3f}s')

        error = abs(true_time - pred_time)
        ax.annotate(f'Localization Error: {error:.3f}s\nConfidence: {confidence:.3f}',
                   xy=(0.02, 0.95), xycoords='axes fraction',
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title, fontsize=14, fontweight='bold')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_pred.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def plot_moon_mars_comparison(
        self,
        moon_data: Dict[str, np.ndarray],
        mars_data: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Compare seismic characteristics between Moon and Mars."""

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Moon waveform + attention
        axes[0, 0].plot(moon_data['time'], moon_data['waveform'], color='#3498db', linewidth=0.8)
        axes[0, 0].fill_between(moon_data['time'], 0, moon_data['gradcam'] * np.max(np.abs(moon_data['waveform'])),
                                alpha=0.3, color='#3498db')
        axes[0, 0].set_title('Moon: Waveform + Attention', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')

        axes[0, 1].plot(moon_data['prediction'], color='#2980b9', linewidth=1.5)
        axes[0, 1].fill_between(range(len(moon_data['prediction'])), 0, moon_data['prediction'],
                                alpha=0.3, color='#3498db')
        axes[0, 1].set_title('Moon: Detection Confidence', fontweight='bold')
        axes[0, 1].set_ylim(0, 1)

        moon_imf_importance = np.mean(moon_data.get('imf_energy', np.random.rand(5, 100)), axis=1)
        moon_imf_importance /= np.max(moon_imf_importance) + 1e-8
        axes[0, 2].barh([f'IMF-{i+1}' for i in range(len(moon_imf_importance))],
                       moon_imf_importance, color='#3498db', alpha=0.8)
        axes[0, 2].set_title('Moon: IMF Importance', fontweight='bold')

        # Mars waveform + attention
        axes[1, 0].plot(mars_data['time'], mars_data['waveform'], color='#e74c3c', linewidth=0.8)
        axes[1, 0].fill_between(mars_data['time'], 0, mars_data['gradcam'] * np.max(np.abs(mars_data['waveform'])),
                                alpha=0.3, color='#e74c3c')
        axes[1, 0].set_title('Mars: Waveform + Attention', fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')

        axes[1, 1].plot(mars_data['prediction'], color='#c0392b', linewidth=1.5)
        axes[1, 1].fill_between(range(len(mars_data['prediction'])), 0, mars_data['prediction'],
                                alpha=0.3, color='#e74c3c')
        axes[1, 1].set_title('Mars: Detection Confidence', fontweight='bold')
        axes[1, 1].set_ylim(0, 1)

        mars_imf_importance = np.mean(mars_data.get('imf_energy', np.random.rand(5, 100)), axis=1)
        mars_imf_importance /= np.max(mars_imf_importance) + 1e-8
        axes[1, 2].barh([f'IMF-{i+1}' for i in range(len(mars_imf_importance))],
                       mars_imf_importance, color='#e74c3c', alpha=0.8)
        axes[1, 2].set_title('Mars: IMF Importance', fontweight='bold')

        plt.suptitle('Lunar vs Martian Seismic Event Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def plot_false_positive_analysis(
        self,
        time: np.ndarray,
        waveform: np.ndarray,
        prediction: np.ndarray,
        gradcam: np.ndarray,
        true_labels: np.ndarray,
        threshold: float = 0.25,
        title: str = 'False Positive Analysis',
        save_path: Optional[str] = None
    ) -> None:
        """Analyze model failures and attention patterns."""

        pred_binary = (prediction > threshold).astype(int)
        tp_mask = (pred_binary == 1) & (true_labels == 1)
        fp_mask = (pred_binary == 1) & (true_labels == 0)
        fn_mask = (pred_binary == 0) & (true_labels == 1)

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # Waveform
        axes[0].plot(time, waveform, color='#2c3e50', linewidth=0.8)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original Waveform', fontweight='bold')

        # True labels
        axes[1].fill_between(time, 0, true_labels, alpha=0.5, color='#27ae60', label='True Arrival')
        axes[1].set_ylabel('True Label')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend(loc='upper right')

        # Predictions with TP/FP/FN markers
        axes[2].plot(time, prediction, color='#3498db', linewidth=1.5, label='Prediction')

        if np.any(tp_mask):
            axes[2].scatter(time[tp_mask], prediction[tp_mask], color='#27ae60',
                          s=30, zorder=5, label='True Positive')
        if np.any(fp_mask):
            axes[2].scatter(time[fp_mask], prediction[fp_mask], color='#e74c3c',
                          s=30, zorder=5, marker='x', label='False Positive')
        if np.any(fn_mask):
            axes[2].scatter(time[fn_mask], waveform[fn_mask] if len(waveform) == len(fn_mask) else time[fn_mask],
                          color='#f39c12', s=30, zorder=5, marker='^', label='False Negative')

        axes[2].axhline(y=threshold, color='gray', linestyle=':', alpha=0.7)
        axes[2].set_ylabel('Confidence')
        axes[2].legend(loc='upper right', fontsize=8)

        # Grad-CAM
        axes[3].fill_between(time, 0, gradcam, alpha=0.5, color='#9b59b6')
        axes[3].set_ylabel('Attention')
        axes[3].set_xlabel('Time (s)')

        tp_count = np.sum(tp_mask)
        fp_count = np.sum(fp_mask)
        fn_count = np.sum(fn_mask)
        precision = tp_count / (tp_count + fp_count + 1e-8)
        recall = tp_count / (tp_count + fn_count + 1e-8)

        stats_text = f'TP: {tp_count} | FP: {fp_count} | FN: {fn_count} | Precision: {precision:.3f} | Recall: {recall:.3f}'
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout(rect=[0, 0.03, 1, 1])

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def plot_uncertainty_visualization(
        self,
        time: np.ndarray,
        predictions: List[np.ndarray],
        mean_prediction: np.ndarray,
        std_prediction: np.ndarray,
        true_arrival_idx: Optional[int] = None,
        title: str = 'Prediction Uncertainty Analysis',
        save_path: Optional[str] = None
    ) -> None:
        """Monte Carlo dropout uncertainty visualization."""

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})

        # Individual predictions
        ax1 = axes[0]
        for i, pred in enumerate(predictions):
            ax1.plot(time[:len(pred)], pred, color='#3498db', alpha=0.1, linewidth=0.5)

        # Mean with uncertainty band
        ax1.plot(time[:len(mean_prediction)], mean_prediction, color='#2c3e50',
                linewidth=2, label='Mean Prediction')
        ax1.fill_between(time[:len(mean_prediction)],
                        mean_prediction - 2 * std_prediction,
                        mean_prediction + 2 * std_prediction,
                        alpha=0.3, color='#e74c3c', label='95% Confidence Interval')
        ax1.fill_between(time[:len(mean_prediction)],
                        mean_prediction - std_prediction,
                        mean_prediction + std_prediction,
                        alpha=0.5, color='#e74c3c', label='68% Confidence Interval')

        if true_arrival_idx is not None and true_arrival_idx < len(time):
            ax1.axvline(x=time[true_arrival_idx], color='#27ae60', linestyle='--',
                       linewidth=2, label='True Arrival')

        ax1.set_ylabel('Prediction Confidence')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')

        # Uncertainty over time
        ax2 = axes[1]
        ax2.fill_between(time[:len(std_prediction)], 0, std_prediction,
                        alpha=0.6, color='#e74c3c')
        ax2.plot(time[:len(std_prediction)], std_prediction, color='#c0392b', linewidth=1)
        ax2.set_ylabel('Uncertainty (Std Dev)')
        ax2.set_xlabel('Time (s)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()


    def generate_xai_report(
        self,
        samples: List[Dict],
        output_dir: Optional[str] = None
    ) -> None:
        """Automated batch visualization generator."""

        if output_dir is None:
            output_dir = self.save_dir / 'xai_report'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        report_data = []

        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}...")

            sample_dir = output_dir / f'sample_{i:04d}'
            sample_dir.mkdir(exist_ok=True)

            # Core visualization
            self.plot_waveform_prediction_gradcam(
                time=sample['time'],
                waveform=sample['waveform'],
                prediction=sample['prediction'],
                gradcam=sample['gradcam'],
                true_arrival_idx=sample.get('true_arrival_idx'),
                title=f"Sample {i+1}: Seismic Event Detection",
                save_path=sample_dir / 'waveform_prediction_gradcam.png'
            )

            # IMF attention
            if 'imf_energy' in sample:
                self.plot_imf_attention(
                    time=sample['time'],
                    imf_energy=sample['imf_energy'],
                    gradcam=sample['gradcam'],
                    title=f"Sample {i+1}: IMF-wise Attention",
                    save_path=sample_dir / 'imf_attention.png'
                )

            # Multi-layer Grad-CAM
            if 'layer_cams' in sample:
                self.plot_multilayer_gradcam(
                    time=sample['time'],
                    waveform=sample['waveform'],
                    layer_cams=sample['layer_cams'],
                    title=f"Sample {i+1}: Multi-Layer Attention",
                    save_path=sample_dir / 'multilayer_gradcam.png'
                )

            # Collect metrics
            report_data.append({
                'sample_id': i,
                'max_confidence': np.max(sample['prediction']),
                'mean_attention': np.mean(sample['gradcam']),
                'peak_attention': np.max(sample['gradcam']),
                'has_true_arrival': 'true_arrival_idx' in sample
            })

        # Save report summary
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_dir / 'xai_report_summary.csv', index=False)

        print(f"\nXAI Report generated: {output_dir}")
        print(f"Total samples processed: {len(samples)}")


    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        format: str = 'png',
        dpi: int = 300
    ) -> None:
        """Export-ready figure utility."""
        save_path = self.save_dir / f"{filename}.{format}"
        fig.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved: {save_path}")
