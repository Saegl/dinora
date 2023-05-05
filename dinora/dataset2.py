import tarfile
import pathlib
from tqdm import tqdm
import requests

from itertools import chain

from torch.utils.data import IterableDataset, DataLoader
from dinora.preprocess_pgn import load_chess_games, chess_positions
from dinora.policy2 import ONE_HOT_ENCODING_EYE

class PGNDataset(IterableDataset):
    """
    Construct pytorch iterable dataset
    from given path strings to chess PGNs

    Example:
    >>> train_data = PGNDataset("path/to/my/train.pgn")
    >>> test_data = PGNDatset("path/to/my/test.pgn")
    """

    def __init__(self, *paths) -> None:
        self.pgn_paths = paths

    def __iter__(self):
        games = chain(*(load_chess_games(path) for path in self.pgn_paths))
        return chess_positions(games)


def download_ccrl_dataset(
    download_folder: pathlib.Path = pathlib.Path("data"),
    chunks_count: int = 250,
):
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
        ccrl_achieve_path.parent.mkdir()
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

    train_paths = (
        extracted_path / f"cclr/train/{i}.pgn" for i in range(1, chunks_count + 1)
    )
    test_paths = (
        extracted_path / f"cclr/test/{i}.pgn" for i in range(1, chunks_count + 1)
    )

    return (
        [PGNDataset(path) for path in train_paths],
        [PGNDataset(path) for path in test_paths],
    )


def random_dataset():
    """
    Dataset with random outputs,
    just to compare different in speed with other datasets
    """
    import numpy as np
    from random import choice

    class RandomDataset(IterableDataset):
        def __iter__(self):
            board = np.random.random((18, 8, 8)).astype(np.float32)
            policy_tensor = choice(ONE_HOT_ENCODING_EYE)

            result = np.random.random(1).astype(np.float32)
            while True:
                yield board, (policy_tensor, result)

    return ([RandomDataset()], [RandomDataset()])
