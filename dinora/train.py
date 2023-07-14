import sys
import json
import logging
import warnings
import pathlib
from datetime import timedelta
from dataclasses import dataclass, asdict
from typing import Literal

import wandb
from wandb.sdk.lib.disabled import RunDisabled

import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from dinora import PROJECT_ROOT
from dinora.datamodules import CompactDataModule
from dinora.models.torchnet.resnet import *
from dinora.train_callbacks import SampleGameGenerator, BoardsEvaluator


warnings.filterwarnings("ignore", message="The dataloader, .* to improve performance.")
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)


@dataclass
class Config:
    matmul_precision: Literal['highest', 'high', 'medium']
    max_time: dict | None
    max_epochs: int # set -1 to ignore

    tune_batch: bool
    batch_size: int  # will be overwritten if tune_batch = True

    tune_learning_rate: bool
    learning_rate: float  # will be overwritten if tune_learning_rate = True
    lr_scheduler_gamma: float  # Multiplicative factor for StepLR
    lr_scheduler_freq: int   # change each steps

    enable_checkpointing: bool
    checkpoint_train_time_interval: dict

    enable_sample_game_generator: bool
    enable_boards_evaluator: bool

    log_every_n_steps: int

    limit_train_batches: int | None
    limit_val_batches: int | None  # FIXME: we use only first batches for validation

    val_check_type: Literal['chunk', 'steps']
    val_check_steps: int  # ignored when type = chunk

    res_channels: int
    res_blocks: int
    policy_channels: int
    value_channels: int
    value_lin_channels: int


def fit(config: Config):
    torch.set_float32_matmul_precision(config.matmul_precision)
    max_time = timedelta(**config.max_time) if config.max_time else None

    def calc_val_check_interval(config: Config) -> int | float:
        if config.val_check_type == 'chunk':
            # 2 ply in move
            # 40 moves on average in game
            # 10k games in each chunk
            positions_in_chunk = 2 * 40 * 10_000
            steps = positions_in_chunk // config.batch_size

            return steps

        elif config.val_check_type == 'steps':
            if config.limit_train_batches:
                return min(config.val_check_steps, config.limit_train_batches)
            else:
                return config.val_check_steps
        else:
            raise ValueError("Enexpected val_check_type")
            
    callbacks = []

    if config.enable_sample_game_generator:
        callbacks.append(SampleGameGenerator())

    if config.enable_boards_evaluator:
        callbacks.append(BoardsEvaluator())

    if config.enable_checkpointing:
        checkpoint_train_time_interval = timedelta(
            **config.checkpoint_train_time_interval
        )
        mc = ModelCheckpoint(
            dirpath=PROJECT_ROOT / "checkpoints/models",
            filename="{epoch}epoch-{step}step",
            save_weights_only=True,
            train_time_interval=checkpoint_train_time_interval,
        )
        callbacks.append(mc)

    model = ResNetLight(
        res_channels=config.res_channels,
        res_blocks=config.res_blocks,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
        value_lin_channels=config.value_lin_channels,

        learning_rate=config.learning_rate,
        lr_scheduler_gamma=config.lr_scheduler_gamma,
        lr_scheduler_freq=config.lr_scheduler_freq
    )

    wandb_logger = WandbLogger(
        project='dinora-chess',
        log_model="all",  # save model weights to wandb
        config={'config_file': asdict(config)},
    )
    
    print("Downloading dataset from wandb")
    dataset_label = 'saegl/dinora-chess/ccrl-compact:latest'
    dataset_artifact = wandb.run.use_artifact(dataset_label)
    dataset_artifact_dir = dataset_artifact.download()
    if isinstance(dataset_artifact_dir, RunDisabled):
        dataset_folder = PROJECT_ROOT / 'data' / 'converted_dataset'
    else:
        dataset_folder = pathlib.Path(dataset_artifact_dir)
    print("Download complete")

    ccrl = CompactDataModule(
        dataset_folder=dataset_folder,
        batch_size=config.batch_size
    )

    # from dinora.datamodules import CCRLDataModule
    # ccrl = CCRLDataModule(
    #     batch_size=config.batch_size,
    # )

    trainer = pl.Trainer(
        max_time=max_time,
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        enable_checkpointing=config.enable_checkpointing,
        default_root_dir=PROJECT_ROOT / "checkpoints",
        val_check_interval=calc_val_check_interval(config),
        callbacks=callbacks,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
    )

    tuner = Tuner(trainer)

    if config.tune_batch:
        tuner.scale_batch_size(model, datamodule=ccrl)
        config.batch_size = ccrl.hparams.batch_size
        trainer.val_check_interval = calc_val_check_interval(config)

    if config.tune_learning_rate:
        tuner.lr_find(model, datamodule=ccrl)

    trainer.fit(
        model=model,
        datamodule=ccrl,
    )


if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except IndexError:
        print("Provide path to config, examples at train_configs/dev.json")
        sys.exit(1)

    with open(path, encoding="utf") as f:
        config = Config(**json.load(f))
    fit(config)
