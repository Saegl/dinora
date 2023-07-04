from itertools import chain

from torch.utils.data import IterableDataset

from dinora.preprocess_pgn import load_state_tensors
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
        states = chain(*(load_state_tensors(path) for path in self.pgn_paths))
        return states


def random_dataset():
    """
    Dataset with random outputs,
    just to compare difference in speed with other datasets
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
