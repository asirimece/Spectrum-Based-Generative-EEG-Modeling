from mne.io import BaseRaw
import pandas as pd
from pathlib import Path
from mne import Annotations


def get_csv_filename(filename: str) -> str:
    csv_filename = filename.split('/')[-1]
    csv_filename = csv_filename.split('.')[0] + '.csv'
    return csv_filename


def get_event_description(event_id: int) -> str:
    if event_id == 0:
        return 'Non-Target Image onset'
    elif event_id == 1:
        return 'Target Image onset'
    elif event_id == 2:
        return 'Start burst or sequence'
    elif event_id == 3:
        return 'End burst or sequence'
    else:
        return 'BAD'


def get_event_name(event_id: int) -> str:
    if event_id == 0:
        return 'E0'
    elif event_id == 1:
        return 'E1'
    elif event_id == 2:
        return 'E2'
    elif event_id == 3:
        return 'E3'
    else:
        return 'BAD'


def get_event_id(description: str) -> int:
    if 'T=0' in description or 'E0' in description:
        return 0
    elif 'T=1' in description or 'E1' in description:
        return 1
    elif 'RSVP_burstSize' in description:
        return 2
    elif 'END' in description:
        return 3
    else:
        return 4


def get_event_info(description: str) -> (int, str, str):
    event_id = get_event_id(description)
    return event_id, get_event_name(event_id), get_event_description(event_id)


def generate_events_df(raw: BaseRaw, source_path: Path | None = None, save_csv: bool = False) -> (pd.DataFrame, str, str):
    events = []
    rate = 0
    burst_size = 0
    burst_id = 0
    idx = 0
    for annotation in raw.annotations:
        onset = annotation['onset']
        duration = annotation['duration']
        description = annotation['description']
        sfreq = raw.info['sfreq']
        if 'RSVP_burstSize' in description:
            rate = int(description.split('rate=')[1].split(' ')[0])
            burst_size = int(description.split('burstSize=')[1].split('_')[0])
            burst_id = int(description.split('block=')[1].split('_')[0])
        onset_sample = int(onset * sfreq)
        target = 1 if 'T=1' in description else 0
        x_coord = int(description.split('x=')[1].split(' ')[0]) if 'x=' in description else 0
        event_id, event_name, event_desc = get_event_info(description)
        events.append(
            [idx, event_id, event_name, event_desc, onset_sample, onset, 0, duration, description, target, x_coord,
             burst_id, burst_size, rate])
        idx += 1
    events_df = pd.DataFrame(events, columns=['idx', 'event_id', 'event_name', 'event_desc', 'onset_sample', 'onset',
                                              'duration', 'orig_duration', 'description', 'target', 'x_coord',
                                              'burst_id', 'burst_size', 'rate'])
    if save_csv and source_path is not None:
        csv_filename = get_csv_filename(source_path.name)
        events_df.to_csv(f'{source_path.parent}/{csv_filename}', index=False)
        return events_df, csv_filename, f'{source_path.parent}/{csv_filename}'
    return events_df, None, None


def get_annotations_mapping(annotations: Annotations):
    description_mapping = {}
    for annotation in annotations:
        description = annotation['description']
        if description not in description_mapping:
            event_id = get_event_id(description)
            description_mapping[description] = get_event_name(event_id)
    return description_mapping


def get_events_mapping() -> dict:
    return {
        'E0': 0,
        'E1': 1,
        'E2': 2,
        'E3': 3
    }
