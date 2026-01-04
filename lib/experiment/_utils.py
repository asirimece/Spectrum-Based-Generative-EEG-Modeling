from mne.channels import make_standard_montage
import random
import randomname
import hashlib
import secrets
import datetime
import numpy as np
import mne
import torch
from mne import EpochsArray

def get_random_hash(length: int) -> str:
    assert 0 < length < 256, "Length must be between 0 and 256."
    random_string = secrets.token_hex(16)
    hash_object = hashlib.sha256()
    hash_object.update(random_string.encode('utf-8'))
    random_hash = hash_object.hexdigest()
    return random_hash[:length]


def get_random_name(name_type: str | None = None) -> str:
    original_state = random.getstate()
    random.seed(None)
    match name_type:
        case "hash":
            name = get_random_hash(8)
        case "date":
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            name = date
        case "number":
            name = str(random.randint(0, 1000))
        case _:
            name = f'{randomname.get_name()}-{random.randint(0, 1000)}'
    random.setstate(original_state)
    return name
