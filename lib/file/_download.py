from pathlib import Path
from tqdm import tqdm
import requests
from requests import Response


def download_file(url: str, local_path: Path):
    response: Response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise ValueError(f"Failed to download file from {url}")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    # Create a progress bar using tqdm
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(local_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    print(f"File {local_path.name} downloaded successfully to: {local_path}")
