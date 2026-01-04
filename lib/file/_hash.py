from pathlib import Path
import hashlib
import os


def hash_folder(folder_path: Path | str) -> str:
    if type(folder_path) is str:
        folder_path = Path(folder_path)

    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    sha256 = hashlib.sha256()

    for folder_name, sub_folders, filenames in os.walk(folder_path):
        for filename in sorted(filenames):
            filepath = os.path.join(folder_name, filename)

            with open(filepath, 'rb') as file:
                # Read the file in chunks to avoid loading the entire file into memory
                for chunk in iter(lambda: file.read(4096), b''):
                    sha256.update(chunk)

    return sha256.hexdigest()


def save_hash_to_file(hash_file_path: Path | str, hash_value: str):
    with open(hash_file_path, 'w') as file:
        file.write(hash_value)


def load_hash_from_file(hash_file_path: Path | str) -> str:
    with open(hash_file_path, 'r') as file:
        return file.read()


def check_hash_folder(folder_path: Path | str, hash_file_path: Path | str) -> bool:
    if type(folder_path) is str:
        folder_path = Path(folder_path)

    if type(hash_file_path) is str:
        hash_file_path = Path(hash_file_path)

    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    if not hash_file_path.exists():
        raise ValueError(f"Hash file {hash_file_path} does not exist")

    return hash_folder(folder_path) == load_hash_from_file(hash_file_path)
