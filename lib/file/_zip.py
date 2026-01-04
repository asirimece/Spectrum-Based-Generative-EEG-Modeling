import zipfile
from pathlib import Path
import os
import shutil


def extract_zip(zip_file: Path, dest_dir: Path, remove_folder: bool = True):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    if remove_folder:
        # Get the original folder name created by the extraction
        original_folder_name = os.path.commonprefix([os.path.join(dest_dir, name) for name in zip_ref.namelist()])

        # Move all files from the original folder to the parent directory
        for item in os.listdir(original_folder_name):
            source_path = os.path.join(original_folder_name, item)
            destination_path = os.path.join(dest_dir, item)
            shutil.move(source_path, destination_path)

        # Remove the original empty folder
        os.rmdir(original_folder_name)

    print(f"Extracted {zip_file.name} to {dest_dir}")
