import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from lib.gen.trainer.utils import get_labels, get_transformed_epochs_per_class
from lib.transform._spectrogram import STFTComputer
from lib.logging import get_logger

logger = get_logger()


class SpectrogramSimilarityEvaluator:
    def __init__(self, config, dataset, gan):
        self.config = config
        self.dataset = dataset
        self.gan = gan
        
        sfreq=128
        params={'enabled': True, 'window_size': 0.5, 'overlap': 0.75, 'n_fft': 256, 'fmin': 1, 'fmax': 40}
        
        self.spectrogram_computer = STFTComputer(sfreq=sfreq, params=params)
        
        storage_path = './work_dirs'
        self.media_path = os.path.join(storage_path, 'AIRPLANE_128_STFT_GEN_CCWGAN', 'media')
        os.makedirs(self.media_path, exist_ok=True)

    def _convert_to_real(self, spectrograms):
        if np.iscomplexobj(spectrograms):
            logger.info("Converting complex spectrograms to real part.")
            return spectrograms.real
        else:
            return spectrograms

    def generate_subsets(self, run, subset_size=100):
        if run.config.test_subset is not None:
            test_run_config = run.config.get_test_config()
            self.dataset.reset(unlock=True)
            self.dataset.load_by_config(test_run_config)
            
            logger.info(f"Dataset loaded for evaluation with subset {run.config.test_subset}")

        apply_post_transforms = True
        if run.config.test_subset is None and run.config.subset_indexes is not None:
            real_subset = get_transformed_epochs_per_class(
                self.dataset,
                indexes=run.config.subset_indexes,
                apply_post_transforms=apply_post_transforms
            )
        else:
            real_subset = get_transformed_epochs_per_class(
                self.dataset,
                num=[('1', subset_size // 2), ('0', subset_size // 2)],
                apply_post_transforms=apply_post_transforms
            )

        real_data = real_subset.get_data()
        
        if len(real_data.shape) == 3:  # If data is not yet spectrograms
            logger.info("Computing STFT")
            real_spectrograms = self.spectrogram_computer.transform(real_data)
        else:
            real_spectrograms = real_data

        real_spectrograms = self._convert_to_real(real_spectrograms)

        n_real_samples = real_spectrograms.shape[0]
        labels_tensor = get_labels(real_subset, dataset_type=self.config.trainer.gen_dataset_type).to(self.config.device)
        z = torch.randn(n_real_samples, self.config.generator.z_dim, 1, 1, device=self.config.device)

        self.gan.generator.eval()
        with torch.no_grad():
            synthetic_subset = self.gan.generator(z, labels_tensor).cpu().numpy()

        return real_spectrograms, synthetic_subset


    def plot_difference_heatmap(self, real, synthetic, title="Difference Heatmap"):
        """Plot a heatmap of the absolute differences between real and synthetic spectrograms"""
        diff = np.abs(real - synthetic).mean(axis=(0, 1))
        
        plt.figure(figsize=(10, 6))
        plt.imshow(diff, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label="Difference")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        
        save_path = os.path.join(self.media_path, "spectrogram_difference_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved Difference Heatmap to {save_path}")


    def plot_overlay(self, real, synthetic, channel_idx=0, title="Overlay of Real and Synthetic"):
        """Overlay real and synthetic spectrograms for a specific channel"""
        # Average over samples
        real_avg = real[:, channel_idx, :, :].mean(axis=0)  
        synthetic_avg = synthetic[:, channel_idx, :, :].mean(axis=0)

        plt.figure(figsize=(12, 6))
        plt.plot(real_avg.mean(axis=0), label="Real", alpha=0.8)
        plt.plot(synthetic_avg.mean(axis=0), label="Synthetic", alpha=0.8)
        plt.title(title)
        plt.legend()
        plt.xlabel("Time Bins")
        plt.ylabel("Amplitude")
        
        save_path = os.path.join(self.media_path, f"spectrogram_overlay_channel_{channel_idx}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved Overlay Plot to {save_path}")


    def evaluate(self, run, subset_size=100, visualize=True):
        """Evaluate similarity metrics between real and synthetic spectrograms"""
        real_subset, synthetic_subset = self.generate_subsets(run, subset_size)

        assert len(real_subset.shape) == 4, "Real subset must be a spectrogram (4D: samples, channels, freqs, time)."
        assert len(synthetic_subset.shape) == 4, "Synthetic subset must be a spectrogram (4D: samples, channels, freqs, time)."

        error = real_subset - synthetic_subset
        mse = np.mean(np.square(error))
        mae = np.mean(np.abs(error))

        correlation = np.corrcoef(real_subset.flatten(), synthetic_subset.flatten())[0, 1]

        real_flat = real_subset.flatten()
        synthetic_flat = synthetic_subset.flatten()
        dot_product = np.dot(real_flat, synthetic_flat)
        cosine_similarity = dot_product / (np.linalg.norm(real_flat) * np.linalg.norm(synthetic_flat) + 1e-10)

        real_abs = np.abs(real_subset)
        synthetic_abs = np.abs(synthetic_subset)
        real_probs = real_abs / (np.sum(real_abs) + 1e-10)
        synthetic_probs = synthetic_abs / (np.sum(synthetic_abs) + 1e-10)
        epsilon = 1e-10
        kl_divergence = np.sum(real_probs * np.log((real_probs + epsilon) / (synthetic_probs + epsilon)))

        if visualize:
            logger.info("Generating visualizations for spectrogram comparison.")
            self.plot_difference_heatmap(real_subset, synthetic_subset, title="Difference Heatmap")
            self.plot_overlay(real_subset, synthetic_subset, title="Overlay of Real and Synthetic")

        return {
            'spectrogram_mse': mse,
            'spectrogram_mae': mae,
            'spectrogram_correlation': correlation,
            'spectrogram_cosine_similarity': cosine_similarity,
            'spectrogram_kl_divergence': kl_divergence
        }
