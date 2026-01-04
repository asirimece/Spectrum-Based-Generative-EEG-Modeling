from typing import Dict, List
import copy


class VisualisationConfig:
    name: str
    label: str
    metric: str
    metric_label: str
    kwargs: Dict[str, str | int | float | bool]

    def __init__(self,
                 name: str,
                 label: str | None,
                 metric: str | None,
                 metric_: str | None,
                 kwargs: Dict[str, str | int | float | bool] | None = None):
        self.name = name
        self.label = label if label is not None else name
        self.metric = metric if metric is not None else 'Accuracy'
        self.metric_label = metric_label if metric_label is not None else 'Accuracy'
        self.kwargs = kwargs if kwargs is not None else {}

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "metric": self.metric,
            "metric_label": self.metric_label,
            "kwargs": self.kwargs
        }


class AugmentationInfo:
    name: str
    kwargs: Dict[str, str | int | float | bool | List[str | int | float | bool]] = {}
    args: List[str | int | float | bool] = []
    optimization_parameter: str | None = None
    op_neutral_value: str | int | float | bool | None = None
    search: None | Dict
    visualisation_config: VisualisationConfig | None = None

    def __init__(self,
                 name: str,
                 kwargs: Dict[str, str | int | float | bool] | List[str | int | float | bool] | None = None,
                 args: List[str | int | float | bool] | None = None,
                 optimization_parameter: str | None = None,
                 op_neutral_value: str | int | float | bool | None = None,
                 search: None | Dict = None,
                 visualisation_config: VisualisationConfig | None = None):
        self.name = name
        self.kwargs = kwargs if kwargs is not None else {}
        self.args = args if args is not None else []
        self.optimization_parameter = optimization_parameter
        self.op_neutral_value = op_neutral_value
        self.search = search
        self.visualisation_config = visualisation_config

    def __eq__(self, other):
        if not isinstance(other, AugmentationInfo):
            return False
        return (self.name == other.name and
                self.optimization_parameter == other.optimization_parameter)

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return {
            "name": self.name,
            "kwargs": self.kwargs,
            "args": self.args,
            "optimization_parameter": self.optimization_parameter,
            "op_neutral_value": self.op_neutral_value,
            "search": self.search,
            "visualisation_config": (self.visualisation_config.to_dict()
                                     if self.visualisation_config is not None else None)
        }
