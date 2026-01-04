from enum import Enum
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import stats
from lib.experiment.evaluation._base import Visualizer
from lib.experiment.evaluation.result import VisualResult, Result, GenResult
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from torchvision.utils import make_grid
import torch
import numpy as np
from numpy import ndarray
import mne
from mne import EpochsArray, EvokedArray
from torch import Tensor
import copy
from typing import Dict
import math


class RunVisualizer(Visualizer):

    def evaluate(self, result: Result, **kwargs) -> VisualResult | List[VisualResult]:
        pass


class VisualRunMetrics(Enum):
    """Enum class for metric types"""
    CONFUSION_MATRIX = 'confusion_matrix'
    ROC_CURVE = 'roc_curve'
    GEN_SYNTHETIC = 'gen_synthetic'
    GEN_REAL = 'gen_real'
    GEN_P300 = 'gen_p300'
    GEN_STATISTICAL = 'gen_statistical'
    GEN_EVOKED_ELECTRODES_TARGET = 'gen_evoked_electrodes_target'
    GEN_EVOKED_ELECTRODES_DISTRACTOR = 'gen_evoked_electrodes_distractor'
    GEN_TOPOMAP = 'gen_topomap'
    GEN_JOINT_EVOKED = 'gen_joint_evoked'
    GEN_PSD = 'gen_psd'
    GEN_PSD_TOPOMAP = 'gen_psd_topomap'
    GEN_IMAGE_MAP = 'gen_image_map'
    GEN_TIME_FREQUENCY = 'gen_time_frequency'
    GEN_DISTRIBUTION = 'gen_distribution'
    GEN_SAMPLE_DISTRIBUTION = 'gen_sample_distribution'
    GEN_STATISTICAL_PSD_ANALYSIS = 'gen_statistical_psd_analysis'


class ConfusionMatrixVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.CONFUSION_MATRIX.value)

    def evaluate(self, data: Result, **kwargs) -> VisualResult:
        if data.labels is None:
            data.labels = ['Non-Target', 'Target']

        # cm = confusion_matrix(data.gt, data.pred, normalize='true')
        cm = confusion_matrix(data.gt, data.pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.labels, yticklabels=data.labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Confusion Matrix')
        plt.close(fig)
        return VisualResult(self.name, ax.get_figure())


class RocCurveVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.ROC_CURVE.value)

    def evaluate(self, data: Result, **kwargs) -> VisualResult:
        n_classes = data.scores.shape[1] if len(data.scores.shape) > 1 else 1

        fig, ax = plt.subplots(figsize=(8, 8))

        for i in range(n_classes):
            class_scores = data.scores[:, i] if n_classes > 1 else data.scores
            fpr, tpr, thresholds = roc_curve(data.gt == i, class_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr, lw=2,
                label=f'Class {i} (AUC = {roc_auc:.2f}'
            )

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        plt.close(fig)
        return VisualResult(self.name, fig)


def create_grid_result(data: ndarray | Tensor, name: str) -> VisualResult:
    data = data if isinstance(data, torch.Tensor) else torch.from_numpy(data)
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    grid = make_grid(data[:32], normalize=True).permute(1, 2, 0).numpy()
    return VisualResult(name, grid)


class GenSyntheticVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_SYNTHETIC.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.synthetic is None:
            raise ValueError("Predictions are required to visualize synthetic data")
        return create_grid_result(data.synthetic.get_data(copy=True), self.name)


class GenRealVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_REAL.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.real is None:
            raise ValueError("Samples are required to visualize real data")
        return create_grid_result(data.real.get_data(copy=True), self.name)


def mean_p300_plot(epochs: EpochsArray, axes, title_prefix: str = ''):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    captions = {0: 'Non-Target', 1: 'Target'}
    epoch_data = epochs.get_data(copy=True)
    for label in np.unique(epochs.events[:, 2]):
        mean = epoch_data[epochs.events[:, 2] == label].mean(axis=(0, 1))
        std = epoch_data[epochs.events[:, 2] == label].std(axis=(0, 1))
        sfreq = epochs.info['sfreq']
        tmin = np.abs(epochs.tmin)
        axes[0].plot(np.arange(len(mean)) / sfreq - tmin, mean, label=captions[label] if label in captions else label)

        axes[1].plot(np.arange(len(mean)) / sfreq - tmin, mean, label=captions[label] if label in captions else label)
        axes[1].fill_between(np.arange(len(mean)) / sfreq - tmin, mean - std, mean + std, alpha=0.25)

    axes[0].legend(fontsize=20)
    for i, title in enumerate((f'{title_prefix} Means', f'{title_prefix} Means with Stds')):
        axes[i].set_title(title, fontsize=20)
        axes[i].set_xlabel('Time (s)')
    return axes, fig


def plot_sample_epochs(
        epochs: EpochsArray,
        axes: List[plt.Axes] | None = None,
        title_prefix: str = ''
):
    captions = {0: 'Non-Target', 1: 'Target'}
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i, ax in enumerate(axes):
        epoch = epochs[f"{i}"][0].get_data(copy=True).squeeze()
        sfreq = epochs.info['sfreq']
        tmin = np.abs(epochs.tmin)
        ax.plot(np.arange(epoch.shape[1]) / sfreq - tmin, epoch.T + np.arange(len(epoch)) * np.max(epoch) * 3)
        ax.set_title(f'{title_prefix} Sample Epoch for Class {captions[i]}')
        ax.set_xlabel('Time (s)')
    return axes, fig


def mne_erp_plot(epochs: EpochsArray, axes: plt.Axes, title_prefix: str = ''):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    captions = {0: 'Non-Target', 1: 'Target'}
    for i, ax in enumerate(axes):
        filtered_epochs = epochs[f"{i}"]
        evoked = filtered_epochs.average()
        evoked.plot(axes=[ax], show=False, gfp=True, spatial_colors=True, unit=False)
        ax.set_title(f'{title_prefix} ERP {captions[i]}')
    return axes, fig


class GenP300Visualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_P300.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None or 'gen_config' not in kwargs:
            raise ValueError("Not all data available to visualize P300")
        fig, axes = plt.subplots(6, 2, figsize=(12, 36))
        axes = axes.flatten()
        axes[:2], _ = mean_p300_plot(data.real, axes[:2], 'Real')
        axes[2:4], _ = mean_p300_plot(data.synthetic, axes[2:4], 'Synthetic')

        axes[4:6], _ = mne_erp_plot(data.real, axes[4:6], 'Real')
        axes[6:8], _ = mne_erp_plot(data.synthetic, axes[6:8], 'Synthetic')

        axes[8:10], _ = plot_sample_epochs(data.real, axes[8:10], 'Real')
        axes[10:12], _ = plot_sample_epochs(data.synthetic, axes[10:12], 'Synthetic')

        return VisualResult(self.name, fig)


def statistical_plot(real: EpochsArray, synthetic: EpochsArray, axes):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    labels = real.events[:, 2]
    captions = {0: 'Non-Target', 1: 'Target'}
    real_data = real.get_data(copy=True)
    synthetic_data = synthetic.get_data(copy=True)
    for i, label in enumerate(np.unique(labels)):
        times_real = real_data[real.events[:, 2] == label].mean(axis=1)
        mean_real = times_real.mean(axis=0)
        times_synthetic = synthetic_data[synthetic.events[:, 2] == label].mean(axis=1)
        mean_synthetic = times_synthetic.mean(axis=0)
        t_stat, p_val = stats.ttest_ind(times_real, times_synthetic, axis=0)

        sfreq = real.info['sfreq']
        tmin = np.abs(real.tmin)

        axes[i].set_title(f'Statistical Analysis {captions[label]}')

        significant = p_val < 0.05
        significant_times = np.where(significant)[0]
        for time in significant_times:
            time_point = time / sfreq - tmin
            axes[i].axvspan(time_point - 0.01, time_point + 0.01, color='lightgrey', alpha=0.5)

        x = np.arange(len(mean_real)) / sfreq - tmin
        axes[i].plot(x, mean_real, label=f"Real {captions[label]}" if label in captions else label)
        axes[i].plot(x, mean_synthetic, label=f"Synthetic {captions[label]}" if label in captions else label)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
    return axes, fig


class GenStatisticalVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_STATISTICAL.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None or 'gen_config' not in kwargs:
            raise ValueError("Not all data available to visualize P300")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes = axes.flatten()
        axes, _ = statistical_plot(data.real, data.synthetic, axes)
        return VisualResult(self.name, fig)


def generate_class_specific_evoked(real: EpochsArray,
                                   synthetic: EpochsArray,
                                   label: str = '0') -> Dict[str, EvokedArray]:
    real = copy.deepcopy(real[label])
    real_evoked = real.average()
    synthetic = copy.deepcopy(synthetic[label])
    synthetic_evoked = synthetic.average()
    delta_evoked = copy.deepcopy(real).subtract_evoked(evoked=synthetic_evoked).average()
    return {
        'delta': delta_evoked,
        'synthetic': synthetic_evoked,
        'real': real_evoked,
    }


def plot_evoked_electrodes(
        real: EpochsArray,
        synthetic: EpochsArray,
        label: str = '0',
        title_prefix: str = '',
        figSize: tuple[int, int] = (16, 10)):
    fig = plt.figure(figsize=figSize)
    captions = {'real': 'Real', 'synthetic': 'Synthetic', 'delta': 'Delta'}
    colors = {'real': 'blue', 'synthetic': 'red', 'delta': 'black'}

    topo_iterator = mne.viz.topo.iter_topography(real.info, fig=fig, layout_scale=0.945,
                                                 fig_facecolor='w', axis_facecolor='w', axis_spinecolor='w')
    evoked_epochs = generate_class_specific_evoked(real, synthetic, label)
    for i, (axis, channel) in enumerate(topo_iterator):
        for name, value in evoked_epochs.items():
            axis.plot(value.times, value.data[channel], color=colors[name], label=captions[name] if i == 0 else None)
        axis.axhline(0, color='black', linewidth=1)
        axis.grid()
        axis.set_title(channel)

    fig.legend(prop={'size': 10}, loc='upper left', bbox_to_anchor=(0, 1))
    fig.suptitle(f"{title_prefix} Evoked Electrodes", fontsize=20)
    return fig


class GenEvokedElectrodesTargetVisualizer(RunVisualizer):
    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_EVOKED_ELECTRODES_TARGET.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.real is None:
            raise ValueError("Not all data available to visualize Evoked Electrodes for Real EEG")
        fig = plot_evoked_electrodes(data.real, data.synthetic, '1', 'Target')
        plt.close(fig)
        return VisualResult(self.name, fig)


class GenEvokedElectrodesDistractorVisualizer(RunVisualizer):
    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_EVOKED_ELECTRODES_DISTRACTOR.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.real is None:
            raise ValueError("Not all data available to visualize Evoked Electrodes for Synthetic EEG")
        fig = plot_evoked_electrodes(data.real, data.synthetic, '0', 'Non-Target')
        plt.close(fig)
        return VisualResult(self.name, fig)


def plot_topomap(epochs: EpochsArray, axes: plt.Axes):
    evoked = epochs.average()
    fig = evoked.plot_topomap(axes=axes, show=False)
    return fig.axes, fig


class GenTopomapVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_TOPOMAP.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize topomap comparison")
        n_times = 6
        n_cols = n_times + 1
        n_rows = 4

        fig = plt.figure(constrained_layout=True, figsize=(12, n_rows * 6))
        fig.suptitle('Topomap Comparison', fontsize=20)

        sub_figures = fig.subfigures(nrows=n_rows, ncols=1)

        epochs = [(copy.deepcopy(data.real['0']), 'Real Non-Target'),
                  (copy.deepcopy(data.synthetic['0']), 'Synthetic Non-Target'),
                  (copy.deepcopy(data.real['1']), 'Real Target'),
                  (copy.deepcopy(data.synthetic['1']), 'Synthetic Target')]

        for row, sub_fig in enumerate(sub_figures):
            epoch = epochs[row]
            sub_fig.suptitle(epoch[1], fontsize=16)
            axes = sub_fig.subplots(1, n_cols)
            axes, _ = plot_topomap(epoch[0], axes)
        plt.close(fig)
        return VisualResult(self.name, fig)


def plot_joint_evoked(epochs: EpochsArray, name: str, title: str) -> VisualResult:
    evoked = epochs.average()
    fig = evoked.plot_joint(show=False)
    fig.suptitle(title, fontsize=20)
    plt.close(fig)
    return VisualResult(name, fig)


class GenJointEvokedVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_JOINT_EVOKED.value)

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize joint evoked plots")
        title = 'Joint Evoked Plot'
        epochs: List[Tuple] = [(copy.deepcopy(data.real['0']),
                                f'{VisualRunMetrics.GEN_JOINT_EVOKED.value}_real_non_target',
                                f'Real Non-Target {title}'),
                               (copy.deepcopy(data.real['1']),
                                f'{VisualRunMetrics.GEN_JOINT_EVOKED.value}_real_target',
                                f'Real Target {title}'),
                               (copy.deepcopy(data.synthetic['0']),
                                f'{VisualRunMetrics.GEN_JOINT_EVOKED.value}_synthetic_non_target',
                                f'Synthetic Non-Target {title}'),
                               (copy.deepcopy(data.synthetic['1']),
                                f'{VisualRunMetrics.GEN_JOINT_EVOKED.value}_synthetic_target',
                                f'Synthetic Target {title}')]

        results: List[VisualResult] = []
        for i, epoch_data in enumerate(epochs):
            results.append(plot_joint_evoked(epoch_data[0], epoch_data[1], epoch_data[2]))

        return results


class GenPSDVisualizer(RunVisualizer):
    fmin: int = 0
    fmax: int = 40
    real_color: str = 'blue'
    synthetic_color: str = 'red'

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_PSD.value)

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize PSD")

        epochs: List[Tuple] = [(copy.deepcopy(data.real['0']), copy.deepcopy(data.synthetic['0']), 0),
                               (copy.deepcopy(data.real['1']), copy.deepcopy(data.synthetic['1']), 1)]

        titles = ['Real Non-Target vs Synthetic Non-Target', 'Real Target vs Synthetic Target', 'Real vs Synthetic']
        keys = ['non_target', 'target', 'all']

        results = []

        for i, (real, synthetic, label) in enumerate(epochs):
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f"PSD Analysis {titles[i]}", fontsize=20)
            axes = axes.flatten()

            real_psd = real.compute_psd(fmin=self.fmin, fmax=self.fmax)
            synthetic_psd = synthetic.compute_psd(fmin=self.fmin, fmax=self.fmax)

            real_psd.plot(axes=axes[0], show=False, color=self.real_color, average=True)
            axes[0].set_title('Real PSD')
            synthetic_psd.plot(axes=axes[1], show=False, color=self.synthetic_color, average=True)
            axes[1].set_title('Synthetic PSD')
            y_lim = axes[0].get_ylim()
            axes[1].set_ylim(y_lim[0], y_lim[1])

            real_psd.plot_topo(axes=axes[2], fig_facecolor='w', axis_facecolor='w', color=self.real_color, show=False)
            axes[2].set_title('Real PSD Topo')
            synthetic_psd.plot_topo(axes=axes[3], fig_facecolor='w',
                                    axis_facecolor='w', color=self.synthetic_color, show=False)
            axes[3].set_title('Synthetic PSD Topo')
            plt.close(fig)
            results.append(VisualResult(f'{self.name}_{keys[i]}', fig))

        return results


class GenPSDTopomapVisualizer(RunVisualizer):
    fmin: int = 0
    fmax: int = 32

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_PSD_TOPOMAP.value)

    def evaluate(self, data: GenResult, **kwargs) -> VisualResult:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize PSD")

        epochs: List[EpochsArray] = [copy.deepcopy(data.real['1']), copy.deepcopy(data.synthetic['1']),
                                     copy.deepcopy(data.real['0']), copy.deepcopy(data.synthetic['0']),
                                     copy.deepcopy(data.real), copy.deepcopy(data.synthetic)]

        titles = ['Real Target', 'Synthetic Target', 'Real Non-Target', 'Synthetic Non-Target', 'Real', 'Synthetic']

        fig, axes = plt.subplots(6, 5, figsize=(12, 20))
        fig.suptitle(f"PSD Topomap Analysis", fontsize=20)
        axes = axes.flatten()

        for i, epoch in enumerate(epochs):
            psd = epoch.compute_psd(fmin=self.fmin, fmax=self.fmax)
            psd.plot_topomap(axes=axes[i * 5:(i + 1) * 5], normalize=True, show=False)
            axes[i * 5].set_title(f'PSD Topomap {titles[i]}', y=1.5)
        plt.close(fig)
        return VisualResult(self.name, fig)


class GenImageMapVisualizer(RunVisualizer):
    # gfp = Global Field Power
    aggregators: List[str | Callable] = ['mean', 'std']
    channels: List[str] | bool

    def __init__(self, aggregators: List[str | Callable] = None, channels: List[str] | bool = False):
        super().__init__(VisualRunMetrics.GEN_IMAGE_MAP.value)
        if aggregators is not None:
            self.aggregators = aggregators
        self.channels = channels

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize Image Map")

        figs = []
        ch_names = data.real.info['ch_names']
        channels: List[str] | None = None
        if isinstance(self.channels, list):
            channels = self.channels
        elif self.channels:
            channels = ch_names

        for agg in self.aggregators:
            real_target = data.real['1'].plot_image(combine=agg, title=f"Real Target {agg}", show=False)[0]
            figs.append(VisualResult(f'{self.name}_real_target_{agg}', real_target))

            synthetic_target = data.synthetic['1'].plot_image(combine=agg, title=f"Synthetic Target {agg}", show=False)[
                0]
            figs.append(VisualResult(f'{self.name}_synthetic_target_{agg}', synthetic_target))
            real_non_target = data.real['0'].plot_image(combine=agg, title=f"Real Non-Target {agg}", show=False)[0]
            figs.append(VisualResult(f'{self.name}_real_non_target_{agg}', real_non_target))

            synthetic_non_target = data.synthetic['0'].plot_image(combine=agg, title=f"Synthetic Non-Target {agg}",
                                                                  show=False)[0]
            figs.append(VisualResult(f'{self.name}_synthetic_non_target_{agg}', synthetic_non_target))

        if channels is not None:
            for ch in self.channels:
                real_target = data.real['1'].plot_image(picks=[ch], title=f"Real Target {ch}", show=False)[0]
                figs.append(VisualResult(f'{self.name}_real_target_{ch}', real_target))

                synthetic_target = data.synthetic['1'].plot_image(picks=[ch],
                                                                  title=f"Synthetic Target {ch}", show=False)[0]
                figs.append(VisualResult(f'{self.name}_synthetic_target_{ch}', synthetic_target))

                real_non_target = data.real['0'].plot_image(picks=[ch], title=f"Real Non-Target {ch}", show=False)[0]
                figs.append(VisualResult(f'{self.name}_real_non_target_{ch}', real_non_target))

                synthetic_non_target = data.synthetic['0'].plot_image(picks=[ch], title=f"Synthetic Non-Target {ch}",
                                                                      show=False)[0]
                figs.append(VisualResult(f'{self.name}_synthetic_non_target_{ch}', synthetic_non_target))

        for result in figs:
            plt.close(result.data)

        return figs


class GenTimeFrequencyVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_TIME_FREQUENCY.value)

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize Time Frequency")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Time Frequency Analysis', fontsize=20)
        axes = axes.flatten()
        freqs = np.arange(2, 32, 2)
        n_cycles = freqs // 2
        real_target_power = data.real['1'].compute_tfr('morlet', freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                       decim=3, average=True)
        real_target_power.plot(combine='mean', show=False, axes=axes[0])
        axes[0].set_title('Real Target')

        synthetic_target_power = data.synthetic['1'].compute_tfr('morlet', freqs=freqs, n_cycles=n_cycles,
                                                                 return_itc=False, decim=3, average=True)
        synthetic_target_power.plot(combine='mean', show=False, axes=axes[1])
        axes[1].set_title('Synthetic Target')

        real_non_target_power = data.real['0'].compute_tfr('morlet', freqs=freqs,
                                                           n_cycles=n_cycles, return_itc=False, decim=3, average=True)
        real_non_target_power.plot(combine='mean', show=False, axes=axes[2])
        axes[2].set_title('Real Non-Target')

        synthetic_non_target_power = data.synthetic['0'].compute_tfr('morlet', freqs=freqs, n_cycles=n_cycles,
                                                                     return_itc=False, decim=3, average=True)
        synthetic_non_target_power.plot(combine='mean', show=False, axes=axes[3])
        axes[3].set_title('Synthetic Non-Target')

        plt.close(fig)
        return [VisualResult(f'{self.name}', fig)]


class GenDistributionVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_DISTRIBUTION.value)

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize Time Frequency")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Distribution Comparison', fontsize=20)
        axes = axes.flatten()

        real_data = data.real.get_data().flatten()
        synthetic_data = data.synthetic.get_data().flatten()
        combined_min = min(real_data.min(), synthetic_data.min())
        combined_max = max(real_data.max(), synthetic_data.max())

        axes[0].hist(real_data, bins=100, color='blue', label='Real')
        axes[0].set_xlim(combined_min, combined_max)
        axes[0].set_title('Real Distribution')

        axes[1].hist(synthetic_data, bins=100, color='red', label='Synthetic')
        axes[1].set_xlim(combined_min, combined_max)
        axes[1].set_title('Synthetic Distribution')
        
        plt.close(fig)
        return [VisualResult(f'{self.name}', fig)]


class GenSampleDistributionVisualizer(RunVisualizer):

    def __init__(self):
        super().__init__(VisualRunMetrics.GEN_SAMPLE_DISTRIBUTION.value)

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize Time Frequency")
        classes = {'0': 'Non_Target', '1': 'Target'}
        figures = []
        for label, caption in classes.items():
            fig, axes = plt.subplots(4, 3, figsize=(12, 8))
            fig.suptitle(f'Sample Distribution Comparison {caption}', fontsize=20)
            axes = axes.flatten()
            for i in range(len(axes)):
                if i >= len(data.synthetic[label].get_data()) or i >= len(data.real[label].get_data()):
                    break
                bins = np.linspace(-3, 3, 100)
                real_sample = data.real[label].get_data()[i].flatten()
                real_hist, _ = np.histogram(real_sample, bins=100)
                axes[i].hist(real_sample, bins=bins, color='blue', label='Real', alpha=0.5)
                synthetic_sample = data.synthetic[label].get_data()[i].flatten()
                synthetic_hist, _ = np.histogram(synthetic_sample, bins=100)
                axes[i].hist(synthetic_sample, bins=bins, color='red', label='Synthetic', alpha=0.5)
                axes[i].legend()
            plt.close(fig)
            figures.append(VisualResult(f'{self.name}_{caption}', fig))
        return figures


def statistical_psd_plot(real: EpochsArray,
                         synthetic: EpochsArray,
                         bands: Dict[str, Tuple[int, int]],
                         title: str) -> plt.Figure:
    n_bands = len(bands)
    n_cols = 2
    n_rows = math.ceil(n_bands / n_cols)
    fig, axes = plt.subplots(nrows=math.ceil(n_bands / n_cols), ncols=n_cols, figsize=(n_cols * 6, n_rows * 5))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.93)
    axes = axes.flatten()
    for i, (band_name, band) in enumerate(bands.items()):
        real_psd = real.compute_psd(fmin=band[0], fmax=band[1])
        synthetic_psd = synthetic.compute_psd(fmin=band[0], fmax=band[1])

        real_psd.plot(average=True, show=False, axes=axes[i], color='b')
        synthetic_psd.plot(average=True, show=False, axes=axes[i], color='r')
        axes[i].collections[0].set_label('Real')
        axes[i].collections[1].set_label('Synthetic')

        real_freqs = real_psd.freqs
        synthetic_freqs = synthetic_psd.freqs
        assert all(real_freqs == synthetic_freqs), 'Frequency mismatch'
        freqs = real_freqs

        real_psd_values = real_psd.get_data()
        synthetic_psd_values = synthetic_psd.get_data()

        t_stats, p_values = stats.ttest_ind(real_psd_values, synthetic_psd_values, axis=0, equal_var=False)

        if len(p_values.shape) == 2:
            freqs = np.tile(freqs, p_values.shape[0])
            p_values = p_values.flatten()
            assert len(freqs) == len(p_values), 'P_values <> Frequency length mismatch'

        axes[i].legend()
        axes[i].set_title(f'{band_name} [{band[0]}-{band[1]} Hz]')
    return fig


class GenStatisticalPSDAnalysisVisualizer(RunVisualizer):

    def __init__(self, bands: Dict[str, Tuple[str, Tuple[int, int]]] | None = None):
        super().__init__(VisualRunMetrics.GEN_STATISTICAL_PSD_ANALYSIS.value)

        if bands is None:
            self.bands = {
                'Delta': (1, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 32),
                'All': (1, 32)
            }

    def evaluate(self, data: GenResult, **kwargs) -> List[VisualResult]:
        figures = []

        if not isinstance(data, GenResult) or data.synthetic is None or data.real is None:
            raise ValueError("Not all data available to visualize statistical psd analysis")

        combined = statistical_psd_plot(data.real, data.synthetic, self.bands, 'Statistical PSD Analysis')
        figures.append(VisualResult(f'{self.name}_combined', combined))

        target = statistical_psd_plot(data.real['1'], data.synthetic['1'], self.bands, 'Statistical PSD Analysis Target')
        figures.append(VisualResult(f'{self.name}_target', target))

        non_target = statistical_psd_plot(data.real['0'], data.synthetic['0'], self.bands,
                                          'Statistical PSD Analysis Non-Target')
        figures.append(VisualResult(f'{self.name}_non_target', non_target))

        return figures


def get_run_visualizers_by_name(name: str, **kwargs) -> RunVisualizer:
    match name:
        case VisualRunMetrics.CONFUSION_MATRIX.value:
            return ConfusionMatrixVisualizer()
        case VisualRunMetrics.ROC_CURVE.value:
            return RocCurveVisualizer()
        case VisualRunMetrics.GEN_SYNTHETIC.value:
            return GenSyntheticVisualizer()
        case VisualRunMetrics.GEN_REAL.value:
            return GenRealVisualizer()
        case VisualRunMetrics.GEN_P300.value:
            return GenP300Visualizer()
        case VisualRunMetrics.GEN_TOPOMAP.value:
            return GenTopomapVisualizer()
        case VisualRunMetrics.GEN_JOINT_EVOKED.value:
            return GenJointEvokedVisualizer()
        case VisualRunMetrics.GEN_EVOKED_ELECTRODES_TARGET.value:
            return GenEvokedElectrodesTargetVisualizer()
        case VisualRunMetrics.GEN_EVOKED_ELECTRODES_DISTRACTOR.value:
            return GenEvokedElectrodesDistractorVisualizer()
        case VisualRunMetrics.GEN_PSD.value:
            return GenPSDVisualizer()
        case VisualRunMetrics.GEN_PSD_TOPOMAP.value:
            return GenPSDTopomapVisualizer()
        case VisualRunMetrics.GEN_STATISTICAL.value:
            return GenStatisticalVisualizer()
        case VisualRunMetrics.GEN_IMAGE_MAP.value:
            return GenImageMapVisualizer(**kwargs)
        case VisualRunMetrics.GEN_TIME_FREQUENCY.value:
            return GenTimeFrequencyVisualizer()
        case VisualRunMetrics.GEN_DISTRIBUTION.value:
            return GenDistributionVisualizer()
        case VisualRunMetrics.GEN_SAMPLE_DISTRIBUTION.value:
            return GenSampleDistributionVisualizer()
        case VisualRunMetrics.GEN_STATISTICAL_PSD_ANALYSIS.value:
            return GenStatisticalPSDAnalysisVisualizer(**kwargs)
        case _:
            raise ValueError(f"Visualizer {name} not found")


def get_run_visualizers(visualizers: List[str | dict]) -> List[RunVisualizer]:
    initialized_visualizers = []
    for visualizer in visualizers:
        if isinstance(visualizer, dict):
            initialized_visualizers.append(get_run_visualizers_by_name(**visualizer))
        else:
            initialized_visualizers.append(get_run_visualizers_by_name(visualizer))
    return initialized_visualizers
