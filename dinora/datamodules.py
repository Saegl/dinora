import json
from typing import Literal
from pathlib import Path

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset

from dinora.ccrl import download_ccrl_dataset
from dinora.dataset import PGNDataset
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


class CompactDataset(TensorDataset):
    def __init__(
            self,
            dataset_folder: Path,
            data: dict[str, int],
            value_type: Literal['scalar', 'wdl'] = 'wdl',
    ) -> None:
        self.dataset_folder = dataset_folder
        self.value_type = value_type
        self.data = data
        self.chunks_bounds = []
        self.length = sum(data.values())

        left_bound = 0
        for rel_path, size in data.items():
            right_bound = left_bound + size

            # [left_bound, right_bound)
            self.chunks_bounds.append({
                'left_bound': left_bound,
                'right_bound': right_bound,
                'path': self.dataset_folder / rel_path
            })

            left_bound = right_bound

        self.current_left_bound = 0
        self.current_right_bound = 0
        self.current_loaded_boards = np.array([])
        self.current_loaded_policies = np.array([])
        self.current_loaded_outcomes = np.array([])
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index):
        if not (self.current_left_bound <= index < self.current_right_bound):
            for chunk_info in self.chunks_bounds:
                if chunk_info['left_bound'] <= index < chunk_info['right_bound']:
                    print('Swith to', chunk_info['path'])

                    data = np.load(chunk_info['path'])
                    self.current_left_bound = chunk_info['left_bound']
                    self.current_right_bound = chunk_info['right_bound']

                    self.current_loaded_boards = data['boards']
                    self.current_loaded_policies = data['policies']
                    self.current_loaded_outcomes = data['outcomes']

                    if self.value_type == 'scalar':
                        self.current_loaded_outcomes = self.current_loaded_outcomes - 1.0
                        self.current_loaded_outcomes = self.current_loaded_outcomes.astype(np.float32).reshape(-1, 1)
                        # TODO: inplace?
                    
                    length = chunk_info['right_bound'] - chunk_info['left_bound']
                    assert length == len(self.current_loaded_boards) \
                        == len(self.current_loaded_outcomes) \
                            == len(self.current_loaded_policies)
                    permutation_index = np.random.permutation(length)

                    self.current_loaded_boards = self.current_loaded_boards[permutation_index]
                    self.current_loaded_policies = self.current_loaded_policies[permutation_index]
                    self.current_loaded_outcomes = self.current_loaded_outcomes[permutation_index]

                    break
            else:
                raise IndexError("Index out of bounds")

        rel_index = index - self.current_left_bound
        board = self.current_loaded_boards[rel_index]
        policy = self.current_loaded_policies[rel_index]
        outcome = self.current_loaded_outcomes[rel_index]
        return compact_state_to_board_tensor(board), (policy, outcome)


class CompactDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_folder: Path,
            batch_size: int = 128,
            value_type: Literal['scalar', 'wdl'] = 'wdl'
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.hparams.batch_size = batch_size
        self.dataset_folder = dataset_folder
        self.value_type = value_type
        report_path = dataset_folder / 'report.json'

        with open(report_path, 'rt', encoding='utf8') as f:
            report = json.load(f)

        self.train_info = report['train']
        self.val_info = report['test']
    
    def train_dataloader(self):
        return DataLoader(
            CompactDataset(self.dataset_folder, self.train_info, self.value_type),
            batch_size=self.batch_size
        )
    
    def val_dataloader(self):
        return DataLoader(
            CompactDataset(self.dataset_folder, self.val_info, self.value_type),
            batch_size=self.batch_size
        )
