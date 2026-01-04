from enum import Enum
from pathlib import Path
from typing import List
from scipy.io import loadmat
import numpy as np
from numpy import ndarray
import mne
from mne.io import BaseRaw, RawArray
from mne.channels import DigMontage
from mne import Info
from itertools import product
import os
from ._eeg import (
    TorchBaseDataset,
    TorchAveragedDataset,
    DatasetType,
    DatasetBuilder,
    build_epochs,
    balance_epochs
)
from datetime import datetime
from lib.utils import format_seconds
from mne import Epochs


class ZhangSubject(Enum):
    SUBJECT_1 = "1"
    SUBJECT_2 = "2"
    SUBJECT_3 = "3"
    SUBJECT_4 = "4"
    SUBJECT_5 = "5"
    SUBJECT_6 = "6"
    SUBJECT_7 = "7"
    SUBJECT_8 = "8"
    SUBJECT_9 = "9"
    SUBJECT_10 = "10"
    SUBJECT_11 = "11"
    SUBJECT_12 = "12"
    SUBJECT_13 = "13"
    SUBJECT_14 = "14"
    SUBJECT_15 = "15"
    SUBJECT_16 = "16"
    SUBJECT_17 = "17"
    SUBJECT_18 = "18"
    SUBJECT_19 = "19"
    SUBJECT_20 = "20"
    SUBJECT_21 = "21"
    SUBJECT_22 = "22"
    SUBJECT_23 = "23"
    SUBJECT_24 = "24"
    SUBJECT_25 = "25"
    SUBJECT_26 = "26"
    SUBJECT_27 = "27"
    SUBJECT_28 = "28"
    SUBJECT_29 = "29"
    SUBJECT_30 = "30"
    SUBJECT_31 = "31"
    SUBJECT_32 = "32"
    SUBJECT_33 = "33"
    SUBJECT_34 = "34"
    SUBJECT_35 = "35"
    SUBJECT_36 = "36"
    SUBJECT_37 = "37"
    SUBJECT_38 = "38"
    SUBJECT_39 = "39"
    SUBJECT_40 = "40"
    SUBJECT_41 = "41"
    SUBJECT_42 = "42"
    SUBJECT_43 = "43"
    SUBJECT_44 = "44"
    SUBJECT_45 = "45"
    SUBJECT_46 = "46"
    SUBJECT_47 = "47"
    SUBJECT_48 = "48"
    SUBJECT_49 = "49"
    SUBJECT_50 = "50"
    SUBJECT_51 = "51"
    SUBJECT_52 = "52"
    SUBJECT_53 = "53"
    SUBJECT_54 = "54"
    SUBJECT_55 = "55"
    SUBJECT_56 = "56"
    SUBJECT_57 = "57"
    SUBJECT_58 = "58"
    SUBJECT_59 = "59"
    SUBJECT_60 = "60"
    SUBJECT_61 = "61"
    SUBJECT_62 = "62"
    SUBJECT_63 = "63"
    SUBJECT_64 = "64"

    def __int__(self) -> int:
        return int(self.value)

    @staticmethod
    def from_string(string: str) -> 'ZhangSubject':
        return ZhangSubject[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'ZhangSubject':
        return ZhangSubject(str(value))


class ZhangGroup(Enum):
    A = "A"
    B = "B"

    def __int__(self) -> int:
        match self:
            case ZhangGroup.A:
                return 0
            case ZhangGroup.B:
                return 1

    @staticmethod
    def from_string(string: str) -> 'ZhangGroup':
        return ZhangGroup[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'ZhangGroup':
        match value:
            case 0:
                return ZhangGroup.A
            case 1:
                return ZhangGroup.B
            case _:
                raise ValueError(f"Unknown group: {value}")


class ZhangBlock(Enum):
    EEGdata1 = 0
    EEGdata2 = 1

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(string: str) -> 'ZhangBlock':
        return ZhangBlock[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'ZhangBlock':
        return ZhangBlock(value)


class ZhangDataset(TorchBaseDataset):
    sfreq: int = 250
    scale_factor: float = 1
    display_rate: int = 10

    def __init__(self, source_path: str | Path | None = None):
        super().__init__("zhang", source_path)

    def _get_filename(
            self,
            subject: ZhangSubject,
            group: ZhangGroup
    ) -> str:
        return f"sub{subject.value}{group.value}.mat"

    def _get_file_path(
            self,
            subjects: List[ZhangSubject | int],
            groups: List[ZhangGroup | int]
    ) -> Path:
        multi_set = len(subjects) > 1 or len(groups) > 1
        subjects = map_int_subjects(subjects)
        groups = map_int_groups(groups)
        if multi_set:
            return Path(self.source_path)
        else:
            subject = subjects[0]
            group = groups[0]
            return Path(self.source_path).joinpath(self._get_filename(subject, group))

    def _get_montage(self) -> DigMontage | None:
        path = self.source_path.joinpath("64-channels.loc")
        try:
            return mne.channels.read_custom_montage(fname=self.source_path.joinpath("64-channels.loc"))
        except Exception as e:
            self.logger.warn(f"Unable to load montage from path: {path}")
            self.logger.warn(e)
            return None

    def _mat_to_raw(self, data: np.ndarray, onsets: ndarray, labels: ndarray) -> RawArray:
        labels[labels == 2] = 0
        data *= self.scale_factor
        montage = self._get_montage()
        info: Info = mne.create_info(
            ch_names=montage.ch_names if montage is not None else [f"eeg_{i}" for i in range(64)],
            ch_types=["eeg"] * data.shape[0],
            sfreq=self.sfreq,
        )
        if montage is not None:
            info.set_montage(montage)
        raw = mne.io.RawArray(data, info)
        annotations1 = mne.Annotations(
            onset=onsets / self.sfreq,
            duration=1 / self.display_rate,
            description=map_labels_to_descriptions(labels),
        )

        raw = raw.set_annotations(annotations1)
        return raw

    def __drop_channels(self, raw: BaseRaw) -> BaseRaw:
        # Mentioned in the dataset paper
        bad_channels = [raw.info.ch_names[32], raw.info.ch_names[42]]
        raw = raw.drop_channels(bad_channels)
        return raw

    def _load_raw_by_filepath(self, file_path: str | Path, blocks: List[ZhangBlock] | None = None) -> mne.io.RawArray:
        if blocks is None:
            blocks = [ZhangBlock.EEGdata1, ZhangBlock.EEGdata2]
        data = loadmat(str(file_path))
        raws = []
        for block in blocks:
            raws.append(self._mat_to_raw(data[block.name],
                                         data["trigger_positions"][block.value],
                                         data["class_labels"][block.value]))
        raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
        return raw

    def _load_raw(self,
                  subject: ZhangSubject,
                  group: ZhangGroup,
                  blocks: List[ZhangBlock] | None = None) -> mne.io.RawArray:
        file_path = self._get_file_path([subject], [group])
        raw = self._load_raw_by_filepath(file_path, blocks)
        return raw

    def _load_subject_group(self,
                            subject: ZhangSubject | int,
                            group: ZhangGroup | int,
                            blocks: List[ZhangBlock | int] | None = None) -> RawArray:
        if type(subject) is int:
            subject = ZhangSubject.from_int(subject)
        if type(group) is int:
            group = ZhangGroup.from_int(group)
        blocks = map_int_blocks(blocks) if blocks is not None else None
        return self._load_raw(subject=subject, group=group, blocks=blocks)

    def __load_subject(self,
                       subject: ZhangSubject | int,
                       groups: List[int | ZhangGroup] | None = None,
                       blocks: List[ZhangBlock | int] | None = None) -> BaseRaw:
        raws = []
        if groups is None:
            groups = [ZhangGroup.A]
        for group in groups:
            raws.append(self._load_subject_group(subject, group, blocks=blocks))
        return mne.concatenate_raws(raws)

    def load_by(self,
                subjects: List[int | ZhangSubject] | None,
                sessions: List[int | ZhangGroup] | None,
                blocks: List[int | ZhangGroup] | None = None,
                force: bool = False) -> BaseRaw:

        new_meta = self._create_meta(subjects=subjects, sessions=sessions, blocks=blocks)
        if new_meta == self.meta and not force:
            return self.raw

        files = list(product(subjects, sessions))
        self.logger.info(f"Loading epochs for {len(subjects)} subjects and {len(sessions)} sessions "
                         f"-> {len(files)} files.")
        self.logger.info("This may take a while...")

        start = datetime.now()
        raws = []
        for i, (subject, group) in enumerate(files):
            self.logger.info(f"Loading subject: {subject}, group: {group} -> {i+1}/{len(files)}")
            raws.append(self._load_subject_group(subject, group, blocks=blocks))
        self.raw = mne.concatenate_raws(raws)
        self.raw = self.__drop_channels(self.raw)
        self.raw_preprocess()
        self.logger.info(f"Loaded {len(raws)} raw files in {format_seconds((datetime.now() - start).total_seconds())}")
        self.meta = new_meta
        return self.raw

    def load(self) -> BaseRaw:
        return self.load_by(subjects=[ZhangSubject.SUBJECT_1], sessions=[ZhangGroup.A])

    def available_subjects(self) -> List[int]:
        return [int(subject) for subject in ZhangSubject]

    def available_groups(self) -> List[str]:
        return [group.value for group in ZhangGroup]

    def available_blocks(self) -> List[int]:
        return [block.value for block in ZhangBlock]

    def load_balanced(self,
                      subjects: List[int | ZhangSubject] | None,
                      sessions: List[int | ZhangGroup] | None,
                      blocks: List[int | ZhangGroup] | None = None) -> Epochs:
        new_meta = self._create_meta(subjects=subjects, sessions=sessions, blocks=blocks)
        if new_meta == self.meta and (self.epochs is not None or self.raw is not None):
            return self.epochs

        files = list(product(subjects, sessions))
        self.logger.info(f"Loading balanced epochs for {len(subjects)} subjects and {len(sessions)} sessions "
                         f"-> {len(files)} files.")
        self.logger.info("This may take a while...")

        final_epochs = []
        for i, (subject, group) in enumerate(files):
            self.logger.info(f"Loading balanced epochs for subject: {subject}, session: {group} -> {i+1}/{len(files)}")
            raw = self._load_subject_group(subject, group, blocks=blocks)
            raw = self.__drop_channels(raw)
            if self.raw_preprocessors is not None and len(self.raw_preprocessors) > 0:
                for preprocessor in self.raw_preprocessors:
                    raw = preprocessor.transform(raw)
            epochs = build_epochs(raw, t_min=self.t_min, t_max=self.t_max)
            balanced_epochs = balance_epochs(epochs, classes=self.dataset_classes)
            final_epochs.append(balanced_epochs)
        self.epochs = mne.concatenate_epochs(final_epochs)
        self.meta = new_meta
        self.reset_raw()


class ZhangAveragedDataset(ZhangDataset, TorchAveragedDataset):

    def __init__(self, source_path: str | Path | None = None):
        super().__init__(source_path)


class ZhangDatasetBuilder(DatasetBuilder):
    download_url: str = "http://bci.med.tsinghua.edu.cn/download.html"
    _dataset_type: DatasetType = DatasetType.BASE

    def __init__(self, local_path: str | Path | None = None):
        if local_path is None:
            local_path = Path(os.getcwd()).joinpath("data", "airplane")
        super().__init__(local_path)
        pass

    def dataset_type(self, value: DatasetType) -> 'ZhangDatasetBuilder':
        self._dataset_type = value
        return self

    def download(self):
        NotImplementedError(f"Automatic download is not supported for Zhang dataset. "
                            f"Please download manually from {self.download_url}."
                            f"Make sure you are in an educational institution network such as the ZHAW network.")
        pass

    def build(self, force_download: bool = False, verify_hash: bool = True) -> ZhangDataset:
        if not self.local_path.exists() or force_download:
            self.download()
        if self._dataset_type == DatasetType.TORCH_STACKED:
            raise ValueError("Torch stacked dataset is not supported for Zhang dataset.")
        elif self._dataset_type == DatasetType.TORCH_AVERAGED:
            return ZhangAveragedDataset(self.local_path)
        else:
            return ZhangDataset(self.local_path)


def map_int_subjects(subjects: List[int | ZhangSubject]) -> List[ZhangSubject]:
    if all(isinstance(subject, ZhangSubject) for subject in subjects):
        return subjects
    return [ZhangSubject.from_int(subject) for subject in subjects]


def map_int_groups(groups: List[int | ZhangGroup]) -> List[ZhangGroup]:
    if all(isinstance(group, ZhangGroup) for group in groups):
        return groups
    return [ZhangGroup.from_int(group) for group in groups]


def map_int_blocks(blocks: List[int | ZhangBlock]) -> List[ZhangBlock]:
    if all(isinstance(block, ZhangBlock) for block in blocks):
        return blocks
    return [ZhangBlock.from_int(block) for block in blocks]


def get_annotations_mapping() -> dict:
    return {
        '0': "E0",
        '1': "E1",
    }


def map_labels_to_descriptions(labels: ndarray) -> ndarray:
    mapping = get_annotations_mapping()
    descriptions = labels.copy().astype(str)
    for key, value in mapping.items():
        descriptions[descriptions == key] = value
    return descriptions
