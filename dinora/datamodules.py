import lightning.pytorch as pl
from torch.utils.data import DataLoader

from dinora.dataset import download_ccrl_dataset, PGNDataset


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
