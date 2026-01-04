from torch import nn
from numpy import ndarray
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from pathlib import Path
from skorch.history import History
from mne import EpochsArray
from torch import Tensor
from lib.logging import get_logger
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from typing import Any

logger = get_logger()


class Result:
    scores: ndarray | None
    pred: ndarray | None
    gt: ndarray | None
    labels: List[str] | ndarray | None
    module: Pipeline | nn.Module | None

    def __init__(
            self,
            scores: ndarray | Tensor | None = None,
            pred: ndarray | Tensor | None = None,
            gt: ndarray | Tensor | None = None,
            labels: List[str] | ndarray | Tensor | None = None,
            module: Pipeline | nn.Module | None = None
    ):
        self.scores = to_ndarray(scores)
        self.pred = to_ndarray(pred)
        self.gt = to_ndarray(gt)
        self.labels = to_ndarray(labels)
        self.module = module

    def to_dict(self):
        return {
            "scores": self.scores,
            "pred": self.pred,
            "gt": self.gt,
            "labels": self.labels,
            "module": self.module
        }


def to_ndarray(data: ndarray | Tensor | None = None) -> ndarray | None:
    if data is None:
        return None
    elif isinstance(data, Tensor):
        return data.detach().cpu().numpy()
    else:
        return data


class VisualResult:
    data: ndarray | plt.Figure
    name: str
    dataframe: DataFrame | None

    def __init__(self, name: str, data: ndarray | plt.Figure, dataframe: DataFrame | None = None):
        self.name = name
        self.data = data
        self.dataframe = dataframe

    def plot(self):
        if isinstance(self.data, plt.Figure):
            self.data.show()
            return
        elif isinstance(self.data, ndarray):
            plt.imshow(self.data)
            plt.show()
            return

    def save(self, path: str | Path):
        if self.data is not None:
            path = Path(path).joinpath("media")
            path.mkdir(parents=True, exist_ok=True)
            filename = path.joinpath(f"{self.name}.png")
            if isinstance(self.data, ndarray):
                plt.imsave(filename, self.data)
            elif isinstance(self.data, plt.Figure):
                self.data.savefig(filename)
            else:
                raise Exception("Unknown data type to save visual result.")
            if self.dataframe is not None:
                self.dataframe.to_csv(path.joinpath(f"{self.name}.csv"))
            return filename
        else:
            return None

    def to_dict(self):
        return {
            "name": self.name,
            "data": self.data
        }

    @staticmethod
    def load(name: str, path: str | Path) -> 'VisualResult':
        path = Path(path)
        if path.exists():
            data = plt.imread(path)
            return VisualResult(name=name, data=data)
        else:
            raise FileNotFoundError(f"Path {path} does not exist")

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'VisualResult':
        if 'name' in data and ('data' in data or 'path' in data):
            if 'path' in data:
                return VisualResult.load(name=data['name'], path=data['path'])
            else:
                return VisualResult(name=data['name'], data=data['data'])
        else:
            raise ValueError("Data does not contain name and data keys")


class EvaluatedResult(Result):
    metrics: Dict[str, any]
    visualizations: List[VisualResult] = []

    def __init__(self,
                 metrics: Dict[str, any] | None = None,
                 visualizations: List[VisualResult] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics if metrics is not None else {}
        self.visualizations = visualizations if visualizations is not None else []

    def visualisations_to_dict(self) -> Dict[str, any]:
        return {visual.name: visual.data for visual in self.visualizations}

    def summary(self):
        logger.info(f"Evaluated Metrics:")
        for key, value in self.metrics.items():
            logger.info(f"{key}: {np.round(value, 4) if isinstance(value, float) else value}")
        logger.info(f"Generated Visualisations:")
        for visual in self.visualizations:
            logger.info(f"{visual.name}")


class NeuralTrainResult(EvaluatedResult):
    history: History | list | None
    model: any

    def __init__(self,
                 history: History | list | None = None,
                 model: any = None,
                 metrics: Dict[str, any] | None = None,
                 **kwargs):
        super().__init__(metrics=metrics, **kwargs)
        self.history = history
        self.model = model


class GenResult(Result):
    synthetic: EpochsArray | None
    real: EpochsArray | None

    def __init__(self,
                 synthetic: EpochsArray | None = None,
                 real: EpochsArray | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.synthetic = synthetic
        self.real = real
