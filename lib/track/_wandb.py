from ._model import Tracker, TrackerMode, TrackerConfig, TrackerTarget
from typing import Dict, Any, List
import wandb
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from lib.experiment.evaluation.experiment import ExperimentMetrics


class WandBTrackerConfig(TrackerConfig):
    project: str

    def __init__(self,
                 project: str,
                 enabled: bool = True,
                 mode: TrackerMode | str = TrackerMode.ONLINE,
                 target: TrackerTarget | str = TrackerTarget.ALL):
        super().__init__(enabled, mode, target)
        self.project = project

    @staticmethod
    def from_config(config: DictConfig) -> 'WandBTrackerConfig':
        return WandBTrackerConfig(
            project=config.project,
            enabled=config.enabled,
            mode=TrackerMode.from_string(config.mode) if isinstance(config.mode, str) else config.mode,
            target=TrackerTarget.from_string(config.target) if isinstance(config.target, str) else config.target
        )


class WandBMetrics:
    simple_metrics: Dict[str, float | int]
    stepped_metrics: Dict[int, Dict[str, float | int]]

    def __init__(self, simple_metrics: Dict[str, float | int], stepped_metrics: Dict[int, Dict[str, float | int]]):
        self.simple_metrics = simple_metrics
        self.stepped_metrics = stepped_metrics


class WandBMetricsPlugin:
    key: str

    def __init__(self, key: str):
        self.key = key

    def find(self, data: Dict[str, Any]) -> Any:
        return data.get(self.key, None)

    def process(self, data: Any) -> WandBMetrics:
        pass

    def run(self, data: Dict[str, Any]) -> WandBMetrics | None:
        data = self.find(data)
        if data is None:
            return None
        else:
            return self.process(data)


class AugmentationTuningEvaluatorPlugin(WandBMetricsPlugin):

    def __init__(self):
        super().__init__(ExperimentMetrics.AUGMENTATION_TUNING.value)

    def process_augmentation(self, data: Dict) -> WandBMetrics:
        simple_metrics = {}
        stepped_metrics = {}
        if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
            sorted_results = sorted(data['results'], key=lambda x: x['value'])
            power_of_ten = get_largest_fraction(sorted_results, 'value')
            for result in sorted_results:
                key = round(result['value'] * (10 ** power_of_ten))
                metrics = result['metrics']
                metrics[result['parameter']] = result['value']
                if key in stepped_metrics:
                    stepped_metrics[key].update(result['metrics'])
                else:
                    stepped_metrics[key] = result['metrics']
        return WandBMetrics(simple_metrics, stepped_metrics)

    def process_augmentation_subject(self, data: Dict) -> WandBMetrics:
        simple_metrics = {}
        stepped_metrics = {}
        if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
            sorted_results = sorted(data['results'], key=lambda x: x['value'])
            power_of_ten = get_largest_fraction(sorted_results, 'value')
            for result in sorted_results:
                key = round(result['value'] * (10 ** power_of_ten))
                metrics = {f"s_{result['subject']}_{name}": value for name, value in result['metrics'].items()}
                metrics[result['parameter']] = result['value']
                if key in stepped_metrics:
                    stepped_metrics[key].update(metrics)
                else:
                    stepped_metrics[key] = metrics
        return WandBMetrics(simple_metrics, stepped_metrics)

    def process(self, data: Dict[str, Any]) -> WandBMetrics | None:
        augmentation = self.process_augmentation(data.get('augmentation', {}))
        augmentation_subject = self.process_augmentation_subject(data.get('augmentation_subject', {}))
        return merge_metrics([augmentation, augmentation_subject])


class ModelTuningBestParamEvaluatorPlugin(WandBMetricsPlugin):

    def __init__(self):
        super().__init__(ExperimentMetrics.MODEL_TUNING_BEST_PARAM.value)

    def process(self, data: Dict[str, Any]) -> WandBMetrics | None:
        simple_metrics = {}
        if 'best_params' in data and isinstance(data['best_params'], dict):
            simple_metrics.update(data['best_params'])
        return WandBMetrics(simple_metrics, {})


class BasicMetricsPlugin(WandBMetricsPlugin):

    def __init__(self):
        super().__init__("basic")

    def run(self, data: Dict[str, Any]) -> WandBMetrics:
        simple_metrics = filter_primitive_metrics(data)
        return WandBMetrics(simple_metrics, {})


class WandBTracker(Tracker):
    config: WandBTrackerConfig
    metric_plugins: List[WandBMetricsPlugin]

    def __init__(self, name: str, config: WandBTrackerConfig):
        super().__init__(name, config)
        self.metric_plugins = [
            BasicMetricsPlugin(),
            AugmentationTuningEvaluatorPlugin(),
            ModelTuningBestParamEvaluatorPlugin()
        ]

    def init(self, name: str, tags: List[str] | None = None, config: Dict[str, Any] | None = None):
        wandb.init(project=self.config.project, name=name, tags=tags, mode=self.config.mode.value, config=config)
        self._initialized = True

    def set_run_configuration(self, config: Dict):
        wandb.config.update(config, allow_val_change=True)

    def log(self, data: Dict[str, Any], step: int | None = None):
        if self.initialized:
            metrics = self.optimize_metrics(data)
            if len(metrics.stepped_metrics) > 0:
                for metric_step, step_metrics in sorted(metrics.stepped_metrics.items()):
                    wandb.log(step_metrics, step=max(metric_step, wandb.run.step))
            if len(metrics.simple_metrics) > 0:
                wandb.log(metrics.simple_metrics, step=step)

    def log_image(self, data: Dict[str, Any], captions: str | List[str] | None = None, step: int | None = None):
        if self.initialized:
            assert captions is None or isinstance(captions, str) or len(captions) == len(data)
            images = {}
            if isinstance(captions, str):
                captions = [captions] * len(data)
            elif captions is None:
                captions = [key for key in data.keys()]
            for i, (key, value) in enumerate(data.items()):
                if value is not None:
                    if isinstance(value, str):
                        value = plt.imread(value)
                    images[key] = wandb.Image(value, caption=captions[i])
            wandb.log(images, step=step)

    def finish(self):
        wandb.finish()
        self._initialized = False

    @staticmethod
    def from_config(config: DictConfig) -> 'WandBTracker':
        return WandBTracker(
            name=config.name,
            config=WandBTrackerConfig.from_config(config)
        )

    def optimize_metrics(self, data: Dict[str, Any]) -> WandBMetrics:
        metrics = []
        if len(self.metric_plugins) == 0:
            return WandBMetrics(data, {})
        for plugin in self.metric_plugins:
            result = plugin.run(data)
            if result is not None:
                metrics.append(result)
        return merge_metrics(metrics)


def filter_primitive_metrics(metrics: Dict[str, Any]):
    return {key: value for key, value in metrics.items() if isinstance(value, (str, int, float))}


def merge_metrics(metrics: List[WandBMetrics]) -> WandBMetrics:
    simple_metrics = {}
    stepped_metrics = {}
    for metric in metrics:
        simple_metrics.update(metric.simple_metrics)
        for step, step_metrics in metric.stepped_metrics.items():
            if step in stepped_metrics:
                stepped_metrics[step].update(step_metrics)
            else:
                stepped_metrics[step] = step_metrics
    return WandBMetrics(simple_metrics, stepped_metrics)


def count_decimal_places(number):
    try:
        decimal_places = len(str(number).split(".")[1])
        return decimal_places
    except IndexError:
        return 0


def get_largest_fraction(data: List[Dict], key) -> int:
    max_fraction_dict = max(data, key=lambda x: count_decimal_places(x.get(key, 0)))
    return count_decimal_places(max_fraction_dict.get(key, 0))
