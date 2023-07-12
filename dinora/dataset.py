from itertools import chain

from torch.utils.data import IterableDataset

from dinora.pgntools import load_state_tensors


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

    @staticmethod
    def tensors_from_path(path):
        with open(path, "r", encoding="utf8", errors="ignore") as pgn:
            return load_state_tensors(pgn)

    def __iter__(self):
        states = chain(*(PGNDataset.tensors_from_path(path) for path in self.pgn_paths))
        return states
