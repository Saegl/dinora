import lightning.pytorch as pl
from torch.utils.data import DataLoader

from dinora.dataset import download_ccrl_dataset, PGNDataset
from dinora.board_representation2 import compact_state_to_board_tensor


class CCRLDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 128):
        super().__init__()
        self.hparams.batch_size = batch_size
        self.save_hyperparameters()
        train_paths, test_paths = download_ccrl_dataset(chunks_count=250)

        self.train_paths = list(train_paths)
        self.test_paths = list(test_paths)
        self.chunks_count = 250

        assert len(self.train_paths) == len(self.test_paths) == self.chunks_count

        self.val_calls = 0

    def train_dataloader(self):
        return DataLoader(
            PGNDataset(*self.train_paths),
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        path = self.test_paths[self.val_calls % self.chunks_count]
        val_dataloader = DataLoader(
            PGNDataset(path),
            batch_size=self.hparams.batch_size
        )
        self.val_calls += 1
        return val_dataloader


from torch.utils.data import IterableDataset

class CompactDataset(IterableDataset):
    def __init__(self, data) -> None:
        self.data = data

    # Iterator[tuple[npf32, tuple[npf32, npf32]]]
    def __iter__(self):
        boards = self.data['boards']
        policies = self.data['policies']
        outcomes = self.data['outcomes']

        for i in range(len(outcomes)):
            yield compact_state_to_board_tensor(boards[i]), (policies[i], outcomes[i])


import numpy as np

class CompactDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size: int = 128) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(
            CompactDataset(np.load(self.train_path)),
            batch_size=self.batch_size
        )
    
    def val_dataloader(self):
        return DataLoader(
            CompactDataset(np.load(self.val_path)),
            batch_size=self.batch_size
        )
