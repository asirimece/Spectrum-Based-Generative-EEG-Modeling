from ._base import Preprocessor
from lib.config import DictInit
from lib.logging import LIBRARY_LOG_LEVEL
from mne.io import BaseRaw
from enum import Enum
from typing import List, Union
from mne import EpochsArray, Epochs
from lib.config import config_to_primitive


class RawPreprocessors(Enum):
    BandPassFilter = "band_pass_filter"
    ReferenceBuilder = "reference_builder"
    NotchFilter = "notch_filter"
    Resample = "resample"
    ChannelSelector = "channel_selector"


class RawPreprocessor(Preprocessor):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name)

    def transform(self, raw: BaseRaw) -> BaseRaw:
        pass


class BandPassFilter(RawPreprocessor):

    def __init__(self, low: float, high: float, method: str = 'fir'):
        super().__init__(RawPreprocessors.BandPassFilter.value)
        self.f_min = low
        self.f_max = high
        self.method = method

    def transform(self, raw: BaseRaw | Epochs) -> Union[BaseRaw, Epochs]:
        raw = raw.filter(l_freq=self.f_min, h_freq=self.f_max, method=self.method, verbose=LIBRARY_LOG_LEVEL)
        return raw


class NotchFilter(RawPreprocessor):

    def __init__(self, freq: float):
        super().__init__(RawPreprocessors.NotchFilter.value)
        self.freq = freq

    def transform(self, raw: BaseRaw) -> BaseRaw:
        if not isinstance(raw, BaseRaw):
            return raw
        raw = raw.notch_filter(freqs=self.freq, verbose=LIBRARY_LOG_LEVEL)
        return raw


class ReferenceBuilder(RawPreprocessor):
    reference: str

    def __init__(self, reference: str = 'average'):
        super().__init__(RawPreprocessors.ReferenceBuilder.value)
        self.reference = reference

    def transform(self, raw: BaseRaw | Epochs) -> Union[BaseRaw, Epochs]:
        raw = raw.set_eeg_reference(self.reference, projection=False, verbose=LIBRARY_LOG_LEVEL)
        return raw


class Resample(RawPreprocessor):
    sfreq: int

    def __init__(self, sfreq: int):
        super().__init__(RawPreprocessors.Resample.value)
        self.sfreq = sfreq

    def transform(self, raw: BaseRaw | Epochs) -> Union[BaseRaw, Epochs]:
        _ = raw.resample(sfreq=self.sfreq, verbose=LIBRARY_LOG_LEVEL)
        return raw


class ChannelSelector(RawPreprocessor):
    channels: List[str]

    def __init__(self, channels: List[str]):
        super().__init__(RawPreprocessors.ChannelSelector.value)
        self.channels = config_to_primitive(channels)

    def transform(self, raw: BaseRaw | Epochs) -> Union[BaseRaw, Epochs]:
        raw = raw.pick_channels(self.channels)
        return raw


def get_raw_preprocessors(configs: List[DictInit]):
    return [get_raw_preprocessor_by_name(config['name'], *config['args'], **config['kwargs']) for config in configs]


def get_raw_preprocessor_by_name(name: str, *args, **kwargs) -> RawPreprocessor:
    match name:
        case RawPreprocessors.BandPassFilter.value:
            return BandPassFilter(*args, **kwargs)
        case RawPreprocessors.ReferenceBuilder.value:
            return ReferenceBuilder(*args, **kwargs)
        case RawPreprocessors.Resample.value:
            return Resample(*args, **kwargs)
        case RawPreprocessors.NotchFilter.value:
            return NotchFilter(*args, **kwargs)
        case RawPreprocessors.ChannelSelector.value:
            return ChannelSelector(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown raw preprocessor name: {name}")
