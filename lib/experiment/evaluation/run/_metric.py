from enum import Enum
from lib.transform._spectrogram import STFTComputer
from lib.experiment.evaluation import Evaluator
from lib.experiment.evaluation.result import Result, GenResult
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, auc, roc_curve
from typing import List, Dict
from lib.config import config_to_primitive
from lib.preprocess import get_raw_preprocessors, RawPreprocessor
import torch
import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr
from dtaidistance import dtw
from omegaconf import DictConfig
import joblib
from mne import EpochsArray
from sklearn.pipeline import Pipeline
from torch import Tensor
import mne


class RunEvaluator(Evaluator):

    def evaluate(self, result: Result) -> dict[str, any]:
        pass


class RunMetrics(Enum):
    """Enum class for metric types"""
    ACCURACY = 'accuracy'
    BALANCED_ACCURACY = 'balanced_accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1'
    UAR = 'uar'
    WAR = 'war'
    ROC_AUC = 'roc_auc'
    AUC = 'auc'
    TIME_SIGNAL_RECONSTRUCTION = 'gen_time_signal_reconstruction'
    GEN_STATISTICS = 'gen_statistics'
    GEN_DTW = 'gen_dtw'
    PSD_SIMILARITY = 'psd_similarity'
    SYNTHETIC_VS_REAL_METRICS = 'synthetic_vs_real_metrics'


class AccuracyEvaluator(RunEvaluator):
    """Accuracy Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.ACCURACY.value)

    def evaluate(self, result: Result) -> dict:
        return {self.name: accuracy_score(result.gt, result.pred)}


class BalancedAccuracyEvaluator(RunEvaluator):
    """Balanced Accuracy Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.BALANCED_ACCURACY.value)

    def evaluate(self, result: Result) -> dict:
        return {self.name: balanced_accuracy_score(result.gt, result.pred)}


class PrecisionEvaluator(RunEvaluator):
    """Precision Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.PRECISION.value)

    def evaluate(self, result: Result) -> dict:
        return {
            self.name: precision_score(result.gt, result.pred),
            f"{self.name}_weighted": precision_score(result.gt, result.pred, average='weighted'),
            f"{self.name}_micro": precision_score(result.gt, result.pred, average='micro'),
            f"{self.name}_macro": precision_score(result.gt, result.pred, average='macro'),
            f"{self.name}_target": precision_score(result.gt, result.pred, pos_label=1),
            f"{self.name}_distractor": precision_score(result.gt, result.pred, pos_label=0)
        }


class RecallEvaluator(RunEvaluator):
    """Recall Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.RECALL.value)

    def evaluate(self, result: Result) -> dict:
        return {
            f"{self.name}": recall_score(result.gt, result.pred),
            f"{self.name}_weighted": recall_score(result.gt, result.pred, average='weighted'),
            f"{self.name}_micro": recall_score(result.gt, result.pred, average='micro'),
            f"{self.name}_macro": recall_score(result.gt, result.pred, average='macro'),
            f"{self.name}_target": recall_score(result.gt, result.pred, pos_label=1),
            f"{self.name}_distractor": recall_score(result.gt, result.pred, pos_label=0)
        }


class F1Evaluator(RunEvaluator):
    """F1 Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.F1.value)

    def evaluate(self, result: Result) -> dict:
        return {
            self.name: f1_score(result.gt, result.pred),
            f"{self.name}_weighted": f1_score(result.gt, result.pred, average='weighted'),
            f"{self.name}_micro": f1_score(result.gt, result.pred, average='micro'),
            f"{self.name}_macro": f1_score(result.gt, result.pred, average='macro'),
            f"{self.name}_target": f1_score(result.gt, result.pred, pos_label=1),
            f"{self.name}_distractor": f1_score(result.gt, result.pred, pos_label=0)
        }


class RocAucEvaluator(RunEvaluator):
    """ROC AUC Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.ROC_AUC.value)

    def evaluate(self, result: Result) -> dict:
        return {self.name: roc_auc_score(result.gt, result.scores[:, 1])}


class AucEvaluator(RunEvaluator):
    """AUC Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.AUC.value)

    def evaluate(self, result: Result) -> dict:
        fpr_target, tpr_target, _ = roc_curve(result.gt, result.scores[:, 1], pos_label=1)
        fpr_distractor, tpr_distractor, _ = roc_curve(result.gt, result.scores[:, 0], pos_label=0)
        return {f"{self.name}_target": auc(fpr_target, tpr_target),
                f"{self.name}_distractor": auc(fpr_distractor, tpr_distractor)}


class UAREvaluator(RunEvaluator):
    "Unweighted Average Recall (UAR) Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.UAR.value)

    def evaluate(self, result: Result) -> dict:
        pred = result.pred
        y = result.gt
        target_right_num = 0
        target_num = 0
        non_target_num = 0
        non_target_right_num = 0
        for idx, label in enumerate(y):
            if label == 1:
                target_num += 1
                if (pred[idx]) == label:
                    target_right_num += 1
            elif label == 0:
                non_target_num += 1
                if (pred[idx]) == label:
                    non_target_right_num += 1
            else:
                print("Illegal label")
                exit()
        score = round((target_right_num / target_num + non_target_right_num / non_target_num) / 2, 4)
        return {self.name: score}


class WAREvaluator(RunEvaluator):
    "Weighted Average Recall (WAR) Evaluator"""

    def __init__(self):
        super().__init__(RunMetrics.WAR.value)

    def evaluate(self, result: Result) -> dict:
        pred = result.pred
        y = result.gt
        sum_num = 0
        correct_num = 0
        for idx, label in enumerate(y):
            sum_num += 1
            if pred[idx] == label:
                correct_num += 1
        score = round(correct_num / sum_num, 4)
        return {self.name: score}


def calc_mse(a: ndarray, b: ndarray, axis: int = 1) -> ndarray:
    return np.mean(np.square(a - b), axis=axis)


def calc_euclidean(a: ndarray, b: ndarray, axis: int = 1) -> ndarray:
    return np.sqrt(np.sum(np.square(a - b), axis=axis))


def calc_correlation(a, b) -> dict[str, list[float]]:
    if a.shape != b.shape:
        raise ValueError("Epoch shapes must be the same.")
    coefficients = []
    p_values = []
    for i in range(a.shape[0]):
        pearson = pearsonr(a[i], b[i])
        coefficients.append(pearson[0])
        p_values.append(pearson[1])

    return {
        'coefficients': coefficients,
        'p_values': p_values
    }


def get_channel_wise_comparison(a: ndarray, b: ndarray, prefix: str = '') -> dict[str, any]:
    euclidean: ndarray = calc_euclidean(a, b, axis=0)
    mse = calc_mse(a, b, axis=0)
    p_correlation: dict[str, list[float]] = calc_correlation(a, b)
    avg_correlation = np.mean(p_correlation['coefficients'])

    a_avg_time = np.mean(a, axis=0)
    b_avg_time = np.mean(b, axis=0)
    mean_avg_diff_time = np.mean(np.abs(a_avg_time - b_avg_time))

    a_avg_channel = np.mean(a, axis=1)
    b_avg_channel = np.mean(b, axis=1)
    mean_avg_diff_channel = np.mean(np.abs(a_avg_channel - b_avg_channel))

    return {
        f'{prefix}euclidean': euclidean.tolist(),
        f'{prefix}euclidean_sum': np.sum(euclidean),
        f'{prefix}euclidean_mean': np.mean(euclidean),
        f'{prefix}euclidean_std': np.std(euclidean),
        f'{prefix}mse': mse.tolist(),
        f'{prefix}mse_sum': np.sum(mse),
        f'{prefix}mse_mean': np.mean(mse),
        f'{prefix}mse_std': np.std(mse),
        f'{prefix}correlation': {'pearson': p_correlation},
        f'{prefix}avg_correlation': avg_correlation,
        f'{prefix}mean_avg_diff_time': mean_avg_diff_time,
        f'{prefix}mean_avg_diff_channel': mean_avg_diff_channel
    }

class ReconstructionEvaluator(RunEvaluator):
    def __init__(self):
        super().__init__(RunMetrics.TIME_SIGNAL_RECONSTRUCTION.value)

    def evaluate(self, result: GenResult, **kwargs) -> dict[str, any]:
        if not isinstance(result, GenResult) or result.synthetic is None or result.real is None:
            raise ValueError("Not all data available to evaluate reconstruction quality.")

        # Extract time-domain data from real and synthetic epochs
        real_epochs = result.real
        synthetic_epochs = result.synthetic

        x_real = real_epochs.get_data()   
        x_fake = synthetic_epochs.get_data() 

        if x_real.shape != x_fake.shape:
            raise ValueError("Shapes of real and synthetic signals do not match.")

        n_epochs, n_channels, n_times = x_real.shape

        # ---- Time-Domain Fidelity Metrics ----
        error_signal = x_real - x_fake
        mse = np.mean(error_signal**2)
        mae = np.mean(np.abs(error_signal))
        rmse = np.sqrt(mse)

        correlations = []
        for ch in range(n_channels):
            real_flat = x_real[:, ch, :].flatten()
            fake_flat = x_fake[:, ch, :].flatten()
            if np.std(real_flat) > 0 and np.std(fake_flat) > 0:
                corr = np.corrcoef(real_flat, fake_flat)[0, 1]
                correlations.append(corr)
        avg_correlation = np.mean(correlations) if correlations else float('nan')

        signal_power = np.mean(x_real**2)
        noise_power = np.mean(error_signal**2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')

        # ---- Frequency-Domain Consistency & Phase Accuracy Metrics ----
        spectral_errors = []
        phase_errors = []
        
        sfreq=128
        params={'enabled': True, 'window_size': 0.5, 'overlap': 0.75, 'n_fft': 256, 'fmin': 1, 'fmax': 40}
        
        spectrogram_computer = STFTComputer(sfreq=sfreq, params=params)
        
        # Use STFTComputer to compute spectrograms for real and synthetic signals
        real_specs = spectrogram_computer.transform(x_real)
        fake_specs = spectrogram_computer.transform(x_fake)

        for epoch in range(n_epochs):
            for ch in range(n_channels):
                Zxx_real = real_specs[epoch, ch]
                Zxx_fake = fake_specs[epoch, ch]
                spectral_errors.append(np.linalg.norm(np.abs(Zxx_real) - np.abs(Zxx_fake)))
                phase_errors.append(np.linalg.norm(np.angle(Zxx_real) - np.angle(Zxx_fake)))

        spectral_error = np.mean(spectral_errors) if spectral_errors else float('nan')
        phase_error = np.mean(phase_errors) if phase_errors else float('nan')

        # Return all computed metrics in a dictionary
        return {
            'reconstruction_mse': mse,
            'reconstruction_mae': mae,
            'reconstruction_rmse': rmse,
            'reconstruction_correlation': avg_correlation,
            'reconstruction_snr': snr,
            'reconstruction_spectral_error': spectral_error,
            'reconstruction_phase_error': phase_error
        }
        

class GenStatisticsEvaluator(RunEvaluator):
    def __init__(self):
        super().__init__(RunMetrics.GEN_STATISTICS.value)

    def evaluate(self, result: GenResult, **kwargs) -> dict[str, any]:
        if (not isinstance(result, GenResult) or result.synthetic is None or result.real is None):
            raise ValueError("Not all data available to evaluate generative statistics.")
        real_target = np.mean(result.real['1'].get_data(copy=True), axis=0)
        synthetic_target = np.mean(result.synthetic['1'].get_data(copy=True), axis=0)

        real_distractor = np.mean(result.real['0'].get_data(copy=True), axis=0)
        synthetic_distractor = np.mean(result.synthetic['0'].get_data(copy=True), axis=0)

        target = get_channel_wise_comparison(real_target, synthetic_target, prefix=f'target_')
        distractor = get_channel_wise_comparison(real_distractor, synthetic_distractor, prefix=f'distractor_')

        return {**target, **distractor}


class GenDtwEvaluator(Evaluator):
    def __init__(self):
        super().__init__('gen_dtw')

    def evaluate(self, result: GenResult, **kwargs) -> dict[str, any]:
        """ Computes DTW distance between real vs. synthetic data, separated by target (label=1) and distractor (label=0) """
        if (not isinstance(result, GenResult) 
            or result.synthetic is None 
            or result.real is None):
            raise ValueError("Not all data available to evaluate generative statistics (need real + synthetic).")

        ch_names = result.real.info['ch_names']
        if len(ch_names) != result.real.get_data(copy=True).shape[1]:
            ch_names = [f'ch_{i}' for i in range(result.real.get_data(copy=True).shape[1])]

        real_target = np.mean(result.real['1'].get_data(copy=True), axis=0)    
        synthetic_target = np.mean(result.synthetic['1'].get_data(copy=True), axis=0)

        real_distractor = np.mean(result.real['0'].get_data(copy=True), axis=0) 
        synthetic_distractor = np.mean(result.synthetic['0'].get_data(copy=True), axis=0)

        # Compute DTW per channel, for target + distractor
        target_dtw_list = []
        distractor_dtw_list = []

        for ch in range(real_target.shape[0]):
            dist = dtw.distance(real_target[ch], synthetic_target[ch])
            target_dtw_list.append(dist)

        for ch in range(real_distractor.shape[0]):
            dist = dtw.distance(real_distractor[ch], synthetic_distractor[ch])
            distractor_dtw_list.append(dist)

        evaluated_result = {}
        for i, ch_name in enumerate(ch_names):
            evaluated_result[f'target_dtw_{ch_name}'] = target_dtw_list[i]
        for i, ch_name in enumerate(ch_names):
            evaluated_result[f'distractor_dtw_{ch_name}'] = distractor_dtw_list[i]

        # Means
        evaluated_result['target_dtw_mean'] = float(np.mean(target_dtw_list))
        evaluated_result['distractor_dtw_mean'] = float(np.mean(distractor_dtw_list))

        return evaluated_result



class PSDSimilarityEvaluator(RunEvaluator):
    fmax: int = 40
    fmin: int = 0

    def __init__(self):
        super().__init__(RunMetrics.PSD_SIMILARITY.value)

    def evaluate(self, result: Result, **kwargs) -> dict:
        if (not isinstance(result, GenResult) or result.synthetic is None or result.real is None):
            raise ValueError("Not all data available to evaluate generative statistics.")

        real_target_psd = result.real['0'].compute_psd(fmin=self.fmin, fmax=self.fmax).average()
        synthetic_target_psd = result.synthetic['0'].compute_psd(fmin=self.fmin, fmax=self.fmax).average()

        real_distractor_psd = result.real['1'].compute_psd(fmin=self.fmin, fmax=self.fmax).average()
        synthetic_distractor_psd = result.synthetic['1'].compute_psd(fmin=self.fmin, fmax=self.fmax).average()

        target_mse = calc_mse(real_target_psd, synthetic_target_psd)
        distractor_mse = calc_mse(real_distractor_psd, synthetic_distractor_psd)

        return {
            f'target_psd_mse': target_mse,
            f'target_psd_mse_sum': np.sum(target_mse),
            f'distractor_psd_mse': distractor_mse,
            f'distractor_psd_mse_sum': np.sum(distractor_mse)
        }
        

class SyntheticVsRealMetricsEvaluator(RunEvaluator):
    model_path: str
    raw_preprocessors: List[RawPreprocessor]
    evaluators: List[RunEvaluator]

    def __init__(self, model_path: str, raw_preprocessors: List[Dict], evaluators: List[str | Dict] | None = None, **kwargs):
        super().__init__(RunMetrics.SYNTHETIC_VS_REAL_METRICS.value)
        self.model_path = model_path
        self.raw_preprocessors = get_raw_preprocessors(raw_preprocessors)

        if evaluators is None:
            evaluators = self.default_evaluators()
        self.evaluators = list(get_run_evaluators(evaluators))

    @staticmethod
    def default_evaluators() -> List[str]:
        return [RunMetrics.ACCURACY.value, RunMetrics.BALANCED_ACCURACY.value, RunMetrics.PRECISION.value,
                RunMetrics.RECALL.value, RunMetrics.F1.value, RunMetrics.ROC_AUC.value, RunMetrics.AUC.value]

    def load_model(self) -> Pipeline:
        model = joblib.load(self.model_path)
        if not isinstance(model, Pipeline):
            raise ValueError("Model must be a pipeline.")
        for key, step in model.named_steps.items():
            if hasattr(step, 'device'):
                step.set_params(device='cuda' if torch.cuda.is_available() else 'cpu')
                if hasattr(step, 'module_'):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    step.module_.to(device)
        return model

    def preprocess(self, epochs: EpochsArray) -> EpochsArray:
        for preprocessor in self.raw_preprocessors:
            epochs = preprocessor.transform(epochs)
        return epochs

    def evaluate_metrics(self, result: Result) -> dict:
        metrics = {}
        for evaluator in self.evaluators:
            metrics.update(evaluator.evaluate(result))
        return metrics

    def get_results(self, epochs: EpochsArray, model: Pipeline, prefix: str) -> dict:
        epochs = self.preprocess(epochs)
        x = epochs.get_data(copy=True)
        labels = epochs.events[:, 2]
        scores = model.predict_proba(x)
        pred = model.predict(x)
        result: Result = Result(gt=epochs.events[:, 2], pred=pred, scores=scores, labels=labels)
        metrics = self.evaluate_metrics(result)
        return {f'{prefix}_{key}': value for key, value in metrics.items()}

    def evaluate(self, result: GenResult, **kwargs) -> dict:
        if not isinstance(result, GenResult) or result.synthetic is None or result.real is None:
            raise ValueError("Not all data available to evaluate generative statistics.")

        model = self.load_model()

        real = self.get_results(result.real, model, 'real')
        synthetic = self.get_results(result.synthetic, model, 'synthetic')

        return {**real, **synthetic}


def get_run_evaluator_by_name(name: str, **kwargs) -> RunEvaluator:
    match name:
        case RunMetrics.ACCURACY.value:
            return AccuracyEvaluator()
        case RunMetrics.BALANCED_ACCURACY.value:
            return BalancedAccuracyEvaluator()
        case RunMetrics.PRECISION.value:
            return PrecisionEvaluator()
        case RunMetrics.RECALL.value:
            return RecallEvaluator()
        case RunMetrics.F1.value:
            return F1Evaluator()
        case RunMetrics.ROC_AUC.value:
            return RocAucEvaluator()
        case RunMetrics.AUC.value:
            return AucEvaluator()
        case RunMetrics.TIME_SIGNAL_RECONSTRUCTION.value:
            return ReconstructionEvaluator()
        case RunMetrics.GEN_STATISTICS.value:
            return GenStatisticsEvaluator()
        case RunMetrics.GEN_DTW.value:
            return GenDtwEvaluator()
        case RunMetrics.PSD_SIMILARITY.value:
            return PSDSimilarityEvaluator()
        case RunMetrics.UAR.value:
            return UAREvaluator()
        case RunMetrics.WAR.value:
            return WAREvaluator()
        case RunMetrics.SYNTHETIC_VS_REAL_METRICS.value:
            return SyntheticVsRealMetricsEvaluator(**kwargs)
        case _:
            raise ValueError(f"Unknown metric: {name}")


def get_run_evaluators(evaluators: List[str | Dict]) -> list[RunEvaluator]:
    for evaluator in evaluators:
        if isinstance(evaluator, str):
            yield get_run_evaluator_by_name(evaluator)
        elif isinstance(evaluator, (dict, DictConfig)):
            evaluator = config_to_primitive(evaluator)
            yield get_run_evaluator_by_name(**evaluator)
        else:
            raise ValueError("Invalid evaluator configuration.")
