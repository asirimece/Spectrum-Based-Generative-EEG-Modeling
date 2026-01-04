import mne
from lib.gen import Training
from lib.gen.model import GenTrainConfig
from lib.dataset import TorchBaseDataset, DatasetType
from typing import List, Tuple
from sklearn.utils import shuffle as shuffle_fn
from mne import EpochsArray
import numpy as np
import torch
from torch import Tensor
from numpy import ndarray
from typing import Union


def get_eta(training: Training, config: GenTrainConfig) -> float:
    if training.result is None:
        return 0.0
    elif training.result.end_time is None:
        training.result.finish()
    current = training.result.epoch
    total = config.trainer.num_epochs
    return training.result.duration * (total - current)


def get_transformed_epochs(dataset: TorchBaseDataset,
                           indexes: List[int] | None = None,
                           event_id: str | None = None,
                           num: int | None = None,
                           apply_post_transforms: bool = True,
                           shuffle: bool = True) -> EpochsArray:
    epochs = dataset.get_transformed_epochs(apply_post_transforms=apply_post_transforms)

    if indexes is not None:
        # Ensure indexes are within bounds
        max_index = len(epochs.events)
        indexes = [idx for idx in indexes if idx < max_index]
        return epochs[indexes]

    if event_id is not None:
        epochs = epochs[event_id]

    if num is not None and num < len(epochs):
        if shuffle:
            idx = sorted(np.random.choice(len(epochs), num, replace=False))
            epochs = epochs[idx]
        else:
            epochs = epochs[:num]

    return epochs

def get_transformed_epochs_per_class(dataset: TorchBaseDataset,
                                     indexes: List[int] | None = None,
                                     num: List[Tuple[str, int | None]] | None = None,
                                     apply_post_transforms: bool = True,
                                     shuffle: bool = True) -> EpochsArray:
    epoch_list = []
    if indexes is not None:
        return get_transformed_epochs(dataset=dataset, indexes=indexes,
                                      apply_post_transforms=apply_post_transforms, shuffle=False)
    for event_id, num in num:
        epochs = get_transformed_epochs(dataset=dataset, event_id=event_id, num=num,
                                        apply_post_transforms=apply_post_transforms, shuffle=shuffle)
        epoch_list.append(epochs)

    print(f"Type of transformed epochs: {type(epochs)}")
    return mne.concatenate_epochs(epoch_list, add_offset=False)

def get_labels(epochs: EpochsArray, dataset_type: DatasetType = DatasetType.TORCH, n_stacks: int = 1) -> Tensor:
    if dataset_type == DatasetType.TORCH_STACKED:
        reduced_labels = []
        labels = epochs.events[:, 2]
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            reduced_labels.extend([label] * (count // n_stacks))
        return torch.LongTensor(reduced_labels)
    else:
        return torch.LongTensor(epochs.events[:, 2])

def get_events(real: EpochsArray,
               labels: Tensor,
               fake: Tensor,
               config: GenTrainConfig,
               dataset_type: DatasetType) -> ndarray:
    real_data = real.get_data()
    
    if real_data.shape[0] != fake.shape[0]:
        if dataset_type == DatasetType.TORCH_STACKED:
            events = []
            n_stacks = config.trainer.n_stacks
            label_counts = labels.unique(return_counts=True)
            for label, count in zip(label_counts[0], label_counts[1]):
                n_samples = count * n_stacks
                n_samples = min(n_samples, real_data.shape[0])
                sub_samples = real[str(label.item())]
                events.extend(sub_samples.events[:n_samples])
            return np.array(events)
            raise ValueError("Real and Fake data have different number of samples in an unsupported case."
                             "Potentially because of a mismatch between the requested sample and dataset size.")
    else:
        return real.events
