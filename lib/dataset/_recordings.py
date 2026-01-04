from enum import Enum
from pathlib import Path
from typing import List
import mne
from mne.io import BaseRaw
from itertools import product
from ._eeg import (
    TorchBaseDataset,
)
from datetime import datetime
from lib.utils import format_seconds


class RecordingSubject(Enum):
    SUBJECT_1 = "1"

    def __int__(self) -> int:
        return int(self.value)

    @staticmethod
    def from_string(string: str) -> 'RecordingSubject':
        return RecordingSubject[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'RecordingSubject':
        return RecordingSubject(str(value))


class RecordingSession(Enum):
    SESSION_0 = "0"
    SESSION_1 = "1"

    def __int__(self) -> int:
        return int(self.value)

    @staticmethod
    def from_string(string: str) -> 'RecordingSession':
        return RecordingSession[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'RecordingSession':
        return RecordingSession(str(value))


class RecordingBlock(Enum):
    BLOCK_0 = 0
    BLOCK_1 = 1

    def __int__(self) -> int:
        return int(self.value)

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(string: str) -> 'RecordingBlock':
        return RecordingBlock[string.upper()]

    @staticmethod
    def from_int(value: int) -> 'RecordingBlock':
        return RecordingBlock(value)


class RecordingDataset(TorchBaseDataset):

    def __init__(self, source_path: str | Path | None = None):
        super().__init__("recording", source_path)

    def _get_filename(
            self,
            subject: RecordingSubject,
            session: RecordingSession,
            block: RecordingBlock
    ) -> str:
        return f"recording_subject_{subject.value}_session_{session.value}_block_{block.value}.fif"

    def _get_file_path(
            self,
            subjects: List[RecordingSubject | int],
            sessions: List[RecordingSession | int],
            blocks: List[RecordingBlock | int]
    ) -> Path:
        multi_set = len(subjects) > 1 or len(sessions) > 1
        subjects = map_recording_subject(subjects)
        sessions = map_recording_session(sessions)
        blocks = map_recording_block(blocks)
        if multi_set:
            return Path(self.source_path)
        else:
            subject = subjects[0]
            session = sessions[0]
            block = blocks[0]
            return Path(self.source_path).joinpath(self._get_filename(subject, session, block))

    def _drop_channels(self, raw: BaseRaw) -> BaseRaw:
        return raw.pick(picks=['eeg'])

    def _load_raw(self, subject: RecordingSubject, session: RecordingSession, block: RecordingBlock):
        file_path = self._get_file_path([subject], [session], [block])
        return mne.io.read_raw_fif(file_path, preload=True, verbose=False)

    def _map_annotations(self, raw: BaseRaw) -> BaseRaw:
        mapping = {
            '0': 'E0',
            '1': 'E1',
            '2': 'E2',
            '3': 'E3'
        }

        raw.annotations.rename(mapping)
        return raw

    def load_by(self,
                subjects: List[int | RecordingSubject] | None,
                sessions: List[int | RecordingSession] | None,
                blocks: List[int | RecordingBlock] | None = None,
                force: bool = False) -> BaseRaw:

        new_meta = self._create_meta(subjects=subjects, sessions=sessions, blocks=blocks)
        if new_meta == self.meta and not force:
            return self.raw

        files = list(product(subjects, sessions, blocks))
        self.logger.info(f"Loading balanced epochs for {len(subjects)} subjects and {len(sessions)} sessions "
                         f"-> {len(files)} files.")
        self.logger.info("This may take a while...")

        start = datetime.now()
        raws = []
        for i, (subject, session, block) in enumerate(files):
            self.logger.info(f"Loading subject: {subject}, session: {session} -> {i + 1}/{len(files)}")
            raws.append(self._load_raw(subject, session, block=block))
        self.raw = mne.concatenate_raws(raws)
        self.raw = self._drop_channels(self.raw)
        self.raw = self._map_annotations(self.raw)
        self.raw_preprocess()
        self.logger.info(f"Loaded {len(raws)} raw files in {format_seconds((datetime.now() - start).total_seconds())}")
        self.meta = new_meta
        return self.raw

    def load(self) -> BaseRaw:
        return self.load_by(subjects=[RecordingSubject.SUBJECT_1], sessions=[RecordingSession.SESSION_0])

    def available_subjects(self) -> List[int]:
        return [int(subject) for subject in RecordingSubject]

    def available_sessions(self) -> List[str]:
        return [session.value for session in RecordingSession]

    def available_blocks(self) -> List[int]:
        return [block.value for block in RecordingBlock]


def map_recording_subject(subjects: List[int | RecordingSubject]) -> List[RecordingSubject]:
    if all(isinstance(subject, RecordingSubject) for subject in subjects):
        return subjects
    return [RecordingSubject.from_int(subject) for subject in subjects]


def map_recording_session(subjects: List[int | RecordingSession]) -> List[RecordingSession]:
    if all(isinstance(subject, RecordingSubject) for subject in subjects):
        return subjects
    return [RecordingSession.from_int(subject) for subject in subjects]


def map_recording_block(subjects: List[int | RecordingBlock]) -> List[RecordingBlock]:
    if all(isinstance(subject, RecordingSubject) for subject in subjects):
        return subjects
    return [RecordingBlock.from_int(subject) for subject in subjects]
