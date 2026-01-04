from ._model import TuningParameterConfig, ModelTuningConfig
import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from typing import List, Dict
from itertools import product
import random


def get_param_values(param: TuningParameterConfig) -> list:
    assert param.strategy is not None, "Strategy must be defined"
    d_type = param.d_type if param.d_type is not None else float
    match param.strategy:
        case 'range':
            return np.arange(start=param.start, stop=param.end + param.step, step=param.step, dtype=d_type).tolist()
        case 'choice':
            return param.values
        case 'loguniform':
            if param.size is None:
                return stats.loguniform(param.start, param.end)
            else:
                return stats.loguniform.rvs(param.start, param.end, size=param.size).tolist()
        case 'logspace':
            if param.size is not None:
                return np.logspace(param.start, param.end, num=param.size).tolist()
            else:
                return np.logspace(np.log10(param.start), np.log10(param.end), num=20).tolist()
        case 'uniform':
            if param.size is None:
                return stats.uniform(param.start, param.end)
            else:
                return stats.uniform.rvs(param.start, param.end, size=param.size).tolist()
        case 'exponential':
            if param.size is None:
                return stats.expon(param.scale)
            else:
                return stats.expon.rvs(param.scale, size=param.size).tolist()
        case _:
            raise ValueError(f"Strategy {param.strategy} is not supported")


def get_params_grid_from_config(config: ModelTuningConfig) -> Dict:
    params = {}
    for param in config.parameters:
        params[param.name] = get_param_values(param)
    return params


def remap_children(config: ModelTuningConfig, hyperparameter_combinations: List[Dict]) -> List[Dict]:
    for i, param in enumerate(config.parameters):
        if param.children is not None:
            for combination in hyperparameter_combinations:
                for child in param.children:
                    value = combination[param.name]
                    combination[child] = value
                del combination[param.name]
    return hyperparameter_combinations


def get_iterations_from_config(config: ModelTuningConfig) -> List[Dict]:
    space = get_params_grid_from_config(config)
    all_combinations = list(product(*space.values()))
    hyperparameter_combinations = [dict(zip(space.keys(), values)) for values in all_combinations]
    hyperparameter_combinations = remap_children(config, hyperparameter_combinations)
    if config.n_iter is not None:
        random.shuffle(hyperparameter_combinations)
        return hyperparameter_combinations[:config.n_iter]
    else:
        return hyperparameter_combinations


def get_tuner(pipeline: Pipeline, config: ModelTuningConfig) -> GridSearchCV | RandomizedSearchCV | Pipeline:
    if config.strategy in ['grid', 'search']:
        params_grid = get_params_grid_from_config(config)
        if config.strategy == 'grid':
            return GridSearchCV(pipeline, params_grid, cv=config.cv, refit=True)
        elif config.strategy == 'search':
            return RandomizedSearchCV(pipeline, params_grid, cv=config.cv, refit=True)
        else:
            return pipeline
    else:
        return pipeline
