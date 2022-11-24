import tarfile
import pathlib
from tqdm import tqdm
import requests

from itertools import chain

from torch.utils.data import IterableDataset, DataLoader
from dinora.preprocess_pgn import load_chess_games, chess_positions


class PGNDataset(IterableDataset):
    """
    Construct pytorch iterable dataset
    from given path strings to chess PGNs
    """

    def __init__(self, *paths) -> None:
        self.pgn_paths = paths

    def __iter__(self):
        games = chain(*(load_chess_games(path) for path in self.pgn_paths))
        return chess_positions(games)


def download_ccrl_dataset(
    download_folder: pathlib.Path = pathlib.Path("data"),
    count: int = 250,
):
    """
    Download CCRL games (2.5M games) and build PGNDataset from them
    More about dataset in leela blog:
    https://lczero.org/blog/2018/09/a-standard-dataset/

    :param path: the path to save the CCRL dataset
    :param count: 1-250 number of chunks, each chunk 10k games
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

    train_paths = (extracted_path / f"cclr/train/{i}.pgn" for i in range(1, count + 1))
    test_paths = (extracted_path / f"cclr/test/{i}.pgn" for i in range(1, count + 1))

    return PGNDataset(*train_paths), PGNDataset(test_paths)
