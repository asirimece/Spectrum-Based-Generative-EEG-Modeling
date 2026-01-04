from lib.dataset import BaseDataset
from lib.experiment.run.base import RunConfig
import numpy as np
from typing import List


def get_gen_real_test_indexes(dataset, config, size: int = 128) -> List[int]:
    if dataset.epochs is None:
        dataset.load_by(subjects=config.subjects, sessions=config.sessions, blocks=config.blocks)

    target_indexes = np.where(dataset.epochs.events[:, 2] == 1)[0]
    non_target_indexes = np.where(dataset.epochs.events[:, 2] == 0)[0]

    size = min(size, len(target_indexes), len(non_target_indexes))

    target_indexes = np.random.choice(target_indexes, size=size, replace=False)
    non_target_indexes = np.random.choice(non_target_indexes, size=size, replace=False)

    combined_indexes = list(target_indexes) + list(non_target_indexes)

    max_index = len(dataset.epochs.events)
    combined_indexes = [idx for idx in combined_indexes if idx < max_index]

    return combined_indexes

    
def get_spectrogram_subset_indexes(dataset, size: int = 128) -> List[int]:
    if dataset.epochs is None:
        dataset.load_by(subjects=dataset.subjects, sessions=dataset.sessions, blocks=dataset.blocks)
    
    target_indexes = np.where(dataset.epochs.events[:, 2] == 1)[0]
    non_target_indexes = np.where(dataset.epochs.events[:, 2] == 0)[0]
    
    target_indexes = np.random.choice(target_indexes, size=size, replace=False) if len(target_indexes) > size else target_indexes
    non_target_indexes = np.random.choice(non_target_indexes, size=size, replace=False) if len(non_target_indexes) > size else non_target_indexes
    
    return list(target_indexes) + list(non_target_indexes)
