import tarfile
import pathlib

import requests
from tqdm import tqdm


def download_ccrl_dataset(
    download_folder: pathlib.Path = pathlib.Path("data"),
    chunks_count: int = 250,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """
    Download CCRL games (2.5M games) and build PGNDataset from them
    More about dataset in leela blog:
    https://lczero.org/blog/2018/09/a-standard-dataset/

    :param path: the path to save the CCRL dataset
    :param count: 1-250 number of chunks, each chunk = 10k games
    """

    ccrl_achieve_path = download_folder / "ccrl.tar.bz2"
    if ccrl_achieve_path.exists():
        print("CCRL already downloaded")
    else:
        print("CCRL is not downloaded, downloading archieve...")
        ccrl_achieve_path.parent.mkdir(parents=True, exist_ok=True)
        url = "http://storage.lczero.org/files/ccrl-pgn.tar.bz2"
        response = requests.get(url, stream=True)
        with open(ccrl_achieve_path, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=4096)):
                handle.write(data)

    extracted_path = download_folder / "ccrl_pgns"

    if extracted_path.exists():
        print("Extracted games found")
    else:
        print("Extracted games not found, extracting...")
        with tarfile.open(ccrl_achieve_path, "r:bz2") as tar:
            tar.extractall(extracted_path)

    train_paths = list(
        extracted_path / f"cclr/train/{i}.pgn" for i in range(1, chunks_count + 1)
    )
    test_paths = list(
        extracted_path / f"cclr/test/{i}.pgn" for i in range(1, chunks_count + 1)
    )

    return train_paths, test_paths
