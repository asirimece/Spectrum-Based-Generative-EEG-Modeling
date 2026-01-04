from lib.dataset import BaseDataset, TorchBaseDataset, DatasetType, TorchStackedDataset
from lib.dataset import AirplaneDatasetBuilder, ZhangDataset, RecordingDataset, ZhangDatasetBuilder
from pathlib import Path
from lib.dataset.torch.transform import TorchTransform


def load_dataset_by_name(name: str, local_path: str | Path | None = None) -> BaseDataset:
    if name == "airplane":
        return AirplaneDatasetBuilder(local_path=local_path).build()
    elif name == "zhang":
        return ZhangDataset(source_path=local_path)
    elif name == "recording":
        return RecordingDataset(source_path=local_path)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def load_torch_dataset_by_name(
        name: str,
        local_path: str | Path | None = None,
        transforms: TorchTransform = None,
        dataset_type: DatasetType = DatasetType.TORCH) -> TorchBaseDataset | TorchStackedDataset:
    dataset: TorchBaseDataset
    if name == "airplane":
        dataset = (AirplaneDatasetBuilder(local_path=local_path)
                   .dataset_type(dataset_type)
                   .build())
    elif name == "zhang":
        dataset = (ZhangDatasetBuilder(local_path=local_path)
                   .dataset_type(dataset_type)
                   .build())
    elif name == "recording":
        dataset = RecordingDataset(source_path=local_path)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    if transforms is not None:
        dataset.transforms = transforms
    return dataset
