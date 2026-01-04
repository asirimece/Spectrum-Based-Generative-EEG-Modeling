from pathlib import Path
import os
from enum import Enum
from typing import List, Set

from mne import Epochs

from lib.file import extract_zip
from lib.exception import NotLoadedError
from mne.io import BaseRaw
from .airplane import generate_events_df, get_annotations_mapping
from pandas import DataFrame
from lib.dataset import build_epochs
import mne
from ._eeg import (
    Meta,
    DatasetBuilder,
    TorchBaseDataset,
    TorchStackedDataset,
    TorchAveragedDataset,
    DatasetType
)
from ..preprocess import RawPreprocessor


class DisplayFrequency(Enum):
    LOW = "5-Hz"
    MEDIUM = "6-Hz"
    HIGH = "10-Hz"

    @staticmethod
    def from_string(string: str) -> 'DisplayFrequency':
        return DisplayFrequency[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'DisplayFrequency':
        match value:
            case 5:
                return DisplayFrequency.LOW
            case 6:
                return DisplayFrequency.MEDIUM
            case 10:
                return DisplayFrequency.HIGH

    def __int__(self) -> int:
        match self:
            case DisplayFrequency.LOW:
                return 5
            case DisplayFrequency.MEDIUM:
                return 6
            case DisplayFrequency.HIGH:
                return 10


class AirplaneSubject(Enum):
    SUBJECT_2 = "02"
    SUBJECT_3 = "03"
    SUBJECT_4 = "04"
    SUBJECT_6 = "06"
    SUBJECT_8 = "08"
    SUBJECT_9 = "09"
    SUBJECT_10 = "10"
    SUBJECT_11 = "11"
    SUBJECT_12 = "12"
    SUBJECT_13 = "13"
    SUBJECT_14 = "14"

    def __int__(self) -> int:
        return int(self.value)

    @staticmethod
    def from_string(string: str) -> 'AirplaneSubject':
        return AirplaneSubject[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'AirplaneSubject':
        value = str(value).zfill(2)
        return AirplaneSubject(str(value))


class AirplaneSequence(Enum):
    A = "a"
    B = "b"

    def __int__(self) -> int:
        match self:
            case AirplaneSequence.A:
                return 0
            case AirplaneSequence.B:
                return 1

    @staticmethod
    def from_string(string: str) -> 'AirplaneSequence':
        return AirplaneSequence[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'AirplaneSequence':
        match value:
            case 0:
                return AirplaneSequence.A
            case 1:
                return AirplaneSequence.B
            case _:
                raise ValueError(f"Unknown group: {value}")


class AirplaneMeta(Meta):
    frequency: DisplayFrequency | None

    def __init__(self,
                 subjects: List[int] | Set[int] | None,
                 sessions: List[int] | Set[int] | None,
                 frequency: DisplayFrequency | None):
        super().__init__(subjects, sessions, None)
        self.frequency = frequency

    def __eq__(self, other):
        return super().__eq__(other) and self.frequency == other.frequency


class AirplaneDataset(TorchBaseDataset):
    file_path: Path
    meta: AirplaneMeta

    __events_df: DataFrame = None

    def __init__(self, source_path: str | Path | None = None):
        super().__init__("airplane", source_path)
        self.meta = AirplaneMeta(None, None, None)

    def __get_filename(
            self,
            frequency: DisplayFrequency = DisplayFrequency.LOW,
            subject: AirplaneSubject = AirplaneSubject.SUBJECT_2,
            sequence: AirplaneSequence = AirplaneSequence.A
    ) -> str:
        return f"rsvp_{frequency.value.replace('-', '')}_{subject.value}{sequence.value}.edf"

    def __get_file_path(
            self,
            frequency: DisplayFrequency,
            subjects: List[AirplaneSubject | int],
            sequences: List[AirplaneSequence | int]
    ) -> Path:
        multi_set = len(subjects) > 1 or len(sequences) > 1
        subjects = map_int_subjects(subjects)
        sequences = map_int_sequences(sequences)
        if multi_set:
            return self.source_path.joinpath(frequency.value)
        else:
            subject = subjects[0]
            sequence = sequences[0]
            return self.source_path.joinpath(frequency.value, self.__get_filename(frequency, subject, sequence))

    def __load_raw(self, frequency: DisplayFrequency, subject: AirplaneSubject, sequence: AirplaneSequence):
        return mne.io.read_raw_edf(self.__get_file_path(frequency, [subject], [sequence]), preload=True, verbose=False)     

    def __drop_status_channel(self):
        status_channels = [ch for ch in self.raw.info.ch_names if 'status' in ch.lower()]
        self.raw = self.raw.drop_channels(status_channels)

    def __map_annotations(self):
        events_df = self.events_df
        assert len(events_df) == len(self.raw.annotations), 'Event count mismatch'
        mapping = get_annotations_mapping(self.raw.annotations)
        self.raw.annotations.rename(mapping, verbose=False)

    def __setup_montage(self):
        try:
            ch_mapping = {name: name.split(" ")[1] for name in self.raw.ch_names}
            print("Renaming channels with mapping: ")
            print(ch_mapping)
            _ = self.raw.rename_channels(ch_mapping)
            print("Channel names after renaming:")
            print(self.raw.ch_names)
            print("Setting montage to standard 10-20 system.")
            montage = mne.channels.make_standard_montage('standard_1020')
            _ = self.raw.set_montage(montage)
        except Exception as e:
            print("Channels already renamed")

    @property
    def events_df(self) -> DataFrame:
        if self.__events_df is None:
            raise NotLoadedError()
        return self.__events_df

    def load(
            self,
            frequency: DisplayFrequency = DisplayFrequency.LOW,
            subject: AirplaneSubject | int = AirplaneSubject.SUBJECT_2,
            sequence: AirplaneSequence | int = AirplaneSequence.A,
            save_events_csv: bool = True
    ) -> BaseRaw:
        return self.load_by(subjects=[subject], sessions=[sequence], frequency=frequency)

    def __load_subject_sequence(self,
                                subject: AirplaneSubject | int,
                                sequence: AirplaneSequence | int,
                                frequency: DisplayFrequency = DisplayFrequency.LOW,
                                ) -> BaseRaw:
        if type(subject) is int:
            subject = AirplaneSubject.from_int(subject)
        if type(sequence) is int:
            sequence = AirplaneSequence.from_int(sequence)
        return self.__load_raw(frequency=frequency, subject=subject, sequence=sequence)

    def __load_subject(self,
                       subject: AirplaneSubject | int,
                       sessions: List[int | AirplaneSequence] | None = None,
                       frequency: DisplayFrequency = DisplayFrequency.LOW) -> BaseRaw:
        raws = []
        if sessions is None:
            sessions = [AirplaneSequence.A]
        for session in sessions:
            raws.append(self.__load_subject_sequence(subject, session, frequency))

        if len(raws) == 1:
            return raws[0]
        else:
            return mne.concatenate_raws(raws)

    def load_by(self,
                subjects: List[int | AirplaneSubject] | None,
                sessions: List[int | AirplaneSequence] | None,
                blocks: List[int | AirplaneSequence] | None = None,
                frequency: DisplayFrequency = DisplayFrequency.LOW,
                force: bool = False) -> BaseRaw:
        new_meta = AirplaneMeta(subjects, sessions, frequency)

        if new_meta == self.meta and not force:
            return self.raw

        multi_set = len(subjects) > 1 or len(sessions) > 1
        self.file_path = self.__get_file_path(frequency, subjects, sessions)
        raws = [self.__load_subject(subject, sessions, frequency) for subject in subjects]
        self.raw = mne.concatenate_raws(raws)
        self.raw_preprocess()
        self.__events_df, _, _ = generate_events_df(self.raw, self.file_path, save_csv=(not multi_set and False))
        self.__drop_status_channel()
        self.__map_annotations()
        self.__setup_montage()
        self.meta = new_meta
        return self.raw

    def _create_meta(self,
                     subjects: List | None = None,
                     sessions: List | None = None,
                     blocks: List | None = None,
                     frequency: DisplayFrequency | None = None) -> AirplaneMeta:
        return AirplaneMeta(subjects, sessions, frequency=frequency)

    def available_subjects(self) -> List[int]:
        return [int(subject) for subject in AirplaneSubject]

    def available_sessions(self) -> List[int]:
        return [int(sequence) for sequence in AirplaneSequence]

    def available_blocks(self) -> List[int]:
        return []

    def load_balanced(self,
                      subjects: List[int] | None,
                      sessions: List[int] | None,
                      blocks: List[int] | None = None) -> Epochs:
        raise NotImplementedError("Balanced loading is not supported for airplane dataset.")

    def load_fractional(self,
                        subjects: List[int | AirplaneSubject] | None,
                        sessions: List[int | AirplaneSequence] | None,
                        blocks: List[int | AirplaneSequence] | None = None,
                        frequency: DisplayFrequency = DisplayFrequency.LOW,
                        train_fraction: float = 0.8,
                        subset: str = 'train',
                        ) -> BaseRaw:

        raws = []
        for subject in subjects:
            subject_raw = self.load_by([subject], sessions, blocks, frequency, force=True)
            original_epochs = build_epochs(subject_raw, t_min=self.t_min, t_max=self.t_max)
            fraction_length = int(len(original_epochs.events[:, 0]) * train_fraction)
            cut_off_time = original_epochs.events[fraction_length][0] / subject_raw.info['sfreq']
            if subset == 'train':
                t_min = 0
                t_max = cut_off_time + abs(self.t_max)
            else:
                t_min = cut_off_time - abs(self.t_min)
                t_max = None
            subject_raw = subject_raw.copy().crop(tmin=t_min, tmax=t_max)
            raws.append(subject_raw)

        self.raw = mne.concatenate_raws(raws)
        self.raw_preprocess()
        return self.raw


class AirplaneStackedDataset(AirplaneDataset, TorchStackedDataset):

    def __init__(self, source_path: str | Path | None = None):
        super().__init__(source_path)


class AirplaneAveragedDataset(AirplaneDataset, TorchAveragedDataset):

    def __init__(self, source_path: str | Path | None = None):
        super().__init__(source_path)


class AirplaneDatasetBuilder(DatasetBuilder):
    download_url: str = "https://physionet.org/static/published-projects/ltrsvp/eeg-signals-from-an-rsvp-task-1.0.0.zip"
    _dataset_type: DatasetType = DatasetType.BASE
    # Change it to a TORCH-compatible dataset type
    #_dataset_type: DatasetType = DatasetType.TORCH
    
    def __init__(self, local_path: str | Path | None = None):
        if local_path is None:
            local_path = Path(os.getcwd()).joinpath("data", "airplane")
        super().__init__(local_path)
        pass

    def dataset_type(self, value: DatasetType) -> 'AirplaneDatasetBuilder':
        self._dataset_type = value
        return self

    def download(self):
        destination_dir = self.local_path.joinpath("origin")
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir.joinpath("airplane.zip")
        # download_file(self.download_url, destination_path)

        if destination_path.exists():
            extract_zip(destination_path, self.local_path)
            os.rmdir(destination_path)

        pass

    def build(self, force_download: bool = False, y_hash: bool = True) -> AirplaneDataset:
        if not self.local_path.exists() or force_download:
            self.download()
        if self._dataset_type == DatasetType.TORCH_STACKED:
            return AirplaneStackedDataset(self.local_path)
        elif self._dataset_type == DatasetType.TORCH_AVERAGED:
            return AirplaneAveragedDataset(self.local_path)
        else:
            return AirplaneDataset(self.local_path)


def map_int_subjects(subjects: List[int | AirplaneSubject]) -> List[AirplaneSubject]:
    if all(isinstance(subject, AirplaneSubject) for subject in subjects):
        return subjects
    return [AirplaneSubject.from_int(subject) for subject in subjects]


def map_int_sequences(sequences: List[int | AirplaneSequence]) -> List[AirplaneSequence]:
    if all(isinstance(sequence, AirplaneSequence) for sequence in sequences):
        return sequences
    return [AirplaneSequence.from_int(sequence) for sequence in sequences]
