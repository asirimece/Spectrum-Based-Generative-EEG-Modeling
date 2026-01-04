from lib.experiment.run.base import Run
from lib.experiment.augmentation import AugmentationInfo, VisualisationConfig
from lib.experiment.evaluation.experiment.utils import (
    aggregate_runs_metrics,
    group_runs_by_index_key,
    group_runs_by_index_keys
)
from typing import List


class MetricResult:
    metrics: dict[str, any]

    def __init__(self, metrics: dict[str, any]):
        self.metrics = metrics

    def to_dict(self):
        return {
            "metrics": self.metrics
        }


class AugLevelMetricResult(MetricResult):
    optimization_parameter: str
    value: float

    def __init__(self, parameter: str, value: float, metrics: dict[str, any]):
        super().__init__(metrics)
        self.optimization_parameter = parameter
        self.value = value

    def to_dict(self):
        parent_dict = super().to_dict()
        child_dict: dict = {
            "parameter": self.optimization_parameter,
            "value": self.value,
        }
        parent_dict.update(child_dict)
        return parent_dict


class AugSubjectLevelMetricResult(AugLevelMetricResult):
    subject: int

    def __init__(self, parameter: str, value: float, metrics: dict[str, any], subject: int):
        super().__init__(parameter, value, metrics)
        self.subject = subject

    def to_dict(self):
        parent_dict = super().to_dict()
        child_dict: dict = {'subject': self.subject}
        parent_dict.update(child_dict)
        return parent_dict


class GenericAugMetricResult(AugLevelMetricResult):
    attributes: dict[str, any]

    def __init__(self, parameter: str, value: float, metrics: dict[str, any], attributes: dict[str, any]):
        super().__init__(parameter, value, metrics)
        self.attributes = attributes

    def to_dict(self):
        parent_dict = super().to_dict()
        parent_dict.update(self.attributes)
        return parent_dict


class AugMetricLevel:
    name: str
    info: AugmentationInfo
    results: List[AugLevelMetricResult | AugSubjectLevelMetricResult | GenericAugMetricResult]

    def __init__(self, name: str,
                 results: List[AugLevelMetricResult | AugSubjectLevelMetricResult],
                 info: AugmentationInfo):
        self.name = name
        self.results = results
        self.info = info

    def to_dict(self):
        return {
            "name": self.name,
            "info": self.info.to_dict() if self.info is not None else None,
            "results": [result.to_dict() for result in self.results]
        }


class AugmentationMetricEvaluation:
    augmentation: AugMetricLevel | None
    augmentation_subject: AugMetricLevel | None

    def __init__(self, augmentation: AugMetricLevel = None, augmentation_subject: AugMetricLevel = None):
        self.augmentation = augmentation
        self.augmentation_subject = augmentation_subject

    def to_dict(self):
        return {
            "augmentation": self.augmentation.to_dict(),
            "augmentation_subject": self.augmentation_subject.to_dict()
        }


def validate_augmentation_runs(runs: list[Run]) -> None:
    augmentation_techniques = [run.config.tuning_augmentation_config for run in runs]
    if len(augmentation_techniques) > 0 and augmentation_techniques[0] is not None:
        names = [aug.name for aug in augmentation_techniques]
        assert len(set(names)) == 1, "All runs should have the same augmentation technique"
        optimisation_parameters = [aug.optimization_parameter for aug in augmentation_techniques]
        assert len(set(optimisation_parameters)) == 1, "All runs should have the same optimisation parameter"
    else:
        raise ValueError("No augmentation techniques found in some runs")


def validate_augmentation_subject_runs(runs: list[Run]) -> None:
    validate_augmentation_runs(runs)
    subjects = [run.config.subjects for run in runs]
    assert len(set(subjects)) == 1, "All runs should have the same subjects"


def generate_augmentation_evaluation(runs: list[Run]) -> AugLevelMetricResult:
    augmentation_config = runs[0].config.tuning_augmentation_config
    return AugLevelMetricResult(
        parameter=augmentation_config.optimization_parameter,
        value=augmentation_config.kwargs[augmentation_config.optimization_parameter],
        metrics=aggregate_runs_metrics(runs)
    )


def generate_generic_augmentation_evaluation(runs: list[Run], attributes: dict[str, any]) -> GenericAugMetricResult:
    augmentation_config = runs[0].config.tuning_augmentation_config
    return GenericAugMetricResult(
        parameter=augmentation_config.optimization_parameter,
        value=augmentation_config.kwargs[augmentation_config.optimization_parameter],
        metrics=aggregate_runs_metrics(runs),
        attributes=attributes
    )


def get_augmentation_level_evaluation(runs: list[Run], subsets: List[str] | None = None) -> AugMetricLevel:
    if subsets is None:
        subsets = ['test']
    grouped = group_runs_by_index_key(runs, 'augmentation', subsets)
    augmentation_techniques: List[AugmentationInfo] = []
    results: List[AugLevelMetricResult] = []
    for key, group_runs in grouped.items():
        validate_augmentation_runs(group_runs)
        augmentation_config = group_runs[0].config.tuning_augmentation_config
        augmentation_techniques.append(augmentation_config)
        results.append(generate_augmentation_evaluation(group_runs))

    assert len(augmentation_techniques) > 0, "No augmentation technique found in some runs"
    assert all(x == augmentation_techniques[0] for x in augmentation_techniques), \
        "All runs should have the same augmentation technique"
    info = augmentation_techniques[0]
    return AugMetricLevel(name=info.name, results=results, info=info)


def get_augmentation_subject_level_evaluation(runs: list[Run], subsets: List[str] | None = None) -> AugMetricLevel:
    if subsets is None:
        subsets = ['test']
    grouped = group_runs_by_index_keys(runs, ['augmentation', 'subject'], subsets)
    augmentation_techniques: List[AugmentationInfo] = []
    results: List[AugSubjectLevelMetricResult] = []
    for augmentation_key, augmentation_group in grouped.items():
        for subject_key, subject_runs in augmentation_group.items():
            validate_augmentation_runs(subject_runs)
            augmentation_config = subject_runs[0].config.tuning_augmentation_config
            augmentation_techniques.append(augmentation_config)
            subjects = subject_runs[0].config.subjects
            subject: int = int("".join(str(item) for item in subjects)) if subjects is not None else 0
            evaluation = generate_augmentation_evaluation(subject_runs)
            evaluation = AugSubjectLevelMetricResult(
                parameter=evaluation.optimization_parameter,
                value=evaluation.value,
                metrics=evaluation.metrics,
                subject=subject
            )
            results.append(evaluation)
    assert len(augmentation_techniques) > 0, "No augmentation technique found in some runs"
    assert all(x == augmentation_techniques[0] for x in augmentation_techniques), \
        "All runs should have the same augmentation technique"
    info = augmentation_techniques[0]
    return AugMetricLevel(name=info.name, results=results, info=info)


def process_group_runs(group_runs, augmentation_techniques, results):
    """
    Recursive function to process nested group runs.

    Args:
        group_runs (dict): The nested dictionary of group runs.
        augmentation_techniques (list): The list to store augmentation configurations.
        results (list): The list to store results of augmentation evaluations.
    """
    if isinstance(group_runs, dict):
        for key, sub_group_runs in group_runs.items():
            process_group_runs(sub_group_runs, augmentation_techniques, results)
    else:
        # Base case: when group_runs is not a dictionary, it's a list of runs
        validate_augmentation_runs(group_runs)
        config = group_runs[0].config
        augmentation_config = config.tuning_augmentation_config
        augmentation_techniques.append(augmentation_config)
        results.append(generate_generic_augmentation_evaluation(group_runs, config.idx))


def get_generic_augmentation_evaluation(runs: list[Run],
                                        keys: List[str],
                                        subsets: List[str] | None = None) -> AugMetricLevel:
    if subsets is None:
        subsets = ['test']
    grouped = group_runs_by_index_keys(runs, keys, subsets)
    augmentation_techniques: List[AugmentationInfo] = []
    results: List[GenericAugMetricResult] = []
    process_group_runs(grouped, augmentation_techniques, results)

    assert len(augmentation_techniques) > 0, "No augmentation technique found in some runs"
    assert all(x == augmentation_techniques[0] for x in augmentation_techniques), \
        "All runs should have the same augmentation technique"
    info = augmentation_techniques[0]
    return AugMetricLevel(name=info.name, results=results, info=info)
