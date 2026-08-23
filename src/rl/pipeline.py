from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta

import torch
import wandb

from dinora.models.alphanet import AlphaNet
from rl.replay_buffer import ReplayBuffer
from rl.selfplay import analyze_pgn, selfplay
from train.callback import Callback
from train.datamodules import CompactDataModule
from train.fit import AlphaNetConfig
from train.train_callbacks import CPLoss
from train.trainer import Trainer
from train.wandb_logger import WandbLogger

WANDB_LOGS_DIR = pathlib.Path("logs/wandb_logs")


@dataclass
class Config:
    opening_noise_moves: int = 15
    dirichlet_alpha: float = 0.3
    noise_fraction: float = 0.25

    cpuct: float = 1.4

    generations: int = 10
    games_per_generation: int = 200
    epochs_per_generation: int = 1
    nodes_per_move: int = 15

    upload_pgn: bool = True
    pgn_chunk_games_size: int = 10_000
    window_games_size: int = 500_000

    batch_size_train: int = 128
    batch_size_selfplay: int = 128
    learning_rate: float = 0.001

    enable_cploss: bool = True
    cploss_label: str = "saegl/dinora-chess/elite_cploss:latest"
    cploss_batch_size: int = 2048
    cploss_positions: int = 3500

    selfplay_log_interval: int = 10 * 60
    selfplay_num_batch_workers: int = 4
    selfplay_cuda_devices: list[str] = field(default_factory=lambda: ["cuda:0"])

    upload_model: bool = True
    model_conf: AlphaNetConfig = AlphaNetConfig(
        res_channels=128,
        res_blocks=7,
        policy_channels=32,
        value_channels=8,
        value_lin_channels=128,
    )

    @staticmethod
    def from_file(filepath: pathlib.Path) -> Config:
        with filepath.open("rt", encoding="utf8") as f:
            data = json.load(f)

        data["model_conf"] = AlphaNetConfig(**data["model_conf"])
        return Config(**data)


def collect_games(config: Config, model: AlphaNet, replay_buffer: ReplayBuffer) -> None:
    print("STAGE: Game collection")
    start_time = time.time()

    selfplay(
        model,
        config.games_per_generation,
        config.nodes_per_move,
        config.batch_size_selfplay,
        config.cpuct,
        config.opening_noise_moves,
        config.dirichlet_alpha,
        config.noise_fraction,
        replay_buffer,
        config.selfplay_log_interval,
        config.selfplay_num_batch_workers,
        config.selfplay_cuda_devices,
    )
    replay_buffer.current_chunk_file.close()
    analyze_pgn(replay_buffer.current_chunk_path)
    replay_buffer.current_chunk_file = replay_buffer.current_chunk_path.open("a")

    print(
        f"STAGE: Game collection took {timedelta(seconds=int(time.time() - start_time))}"
    )


def fit(
    config: Config,
    model: AlphaNet,
    datamodule: CompactDataModule,
    generation_output_dir: pathlib.Path,
    callbacks: list[Callback],
) -> None:
    print("STAGE: Fit")
    start_time = time.time()
    wandb_logger = WandbLogger(project="dinora-chess")
    trainer = Trainer(
        max_epochs=config.epochs_per_generation,
        logger=wandb_logger,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    model_file = generation_output_dir / "model.ckpt"
    torch.save(model, model_file)

    model.to("cuda")
    # There is no validation dataset in RL
    # trigger validation callbacks manually
    for callback in callbacks:
        callback.on_validation_end(trainer, model)

    if config.upload_model:
        artifact = wandb.Artifact("rl_model", type="rl_model")
        artifact.add_file(str(model_file.absolute()))
        wandb.log_artifact(artifact)

    print(f"STAGE: Fit took {timedelta(seconds=int(time.time() - start_time))}")


def start_rl(config: Config) -> None:
    run = wandb.init(
        job_type="rl",
        project="dinora-chess",
        config=asdict(config),
        dir=WANDB_LOGS_DIR,
    )

    model = AlphaNet(
        filters=config.model_conf.res_channels,
        res_blocks=config.model_conf.res_blocks,
        policy_channels=config.model_conf.policy_channels,
        value_channels=config.model_conf.value_channels,
        value_fc_hidden=config.model_conf.value_lin_channels,
        learning_rate=config.learning_rate,
    ).to("cuda")

    callbacks: list[Callback] = []
    if config.enable_cploss:
        cploss = CPLoss(
            config.cploss_label,
            config.cploss_positions,
            config.cploss_batch_size,
        )
        callbacks.append(cploss)

        wandb_logger = WandbLogger(project="dinora-chess")
        log_trainer = Trainer(
            logger=wandb_logger,
            enable_checkpointing=False,
        )
        cploss.on_validation_end(log_trainer, model)

    output_dir = pathlib.Path.cwd() / "data" / "rl_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_buffer_dir = output_dir / "replay_buffer"
    replay_buffer_dir.mkdir(exist_ok=True)

    replay_buffer = ReplayBuffer(
        replay_buffer_dir,
        pgn_chunk_games_size=config.pgn_chunk_games_size,
        window_games_size=config.window_games_size,
        batch_size=config.batch_size_train,
        upload_pgn=config.upload_pgn,
    )

    model_file = output_dir / "model_init.ckpt"
    torch.save(model, model_file)

    if config.upload_model:
        artifact = wandb.Artifact("rl_model", type="rl_model")
        artifact.add_file(str(model_file.absolute()))
        wandb.log_artifact(artifact)

    for generation in range(config.generations):
        generation_output_dir = output_dir / f"generation-{generation}"
        generation_output_dir.mkdir(exist_ok=True)

        collect_games(config, model, replay_buffer)

        datamodule = replay_buffer.prepare_dataset()
        fit(config, model, datamodule, generation_output_dir, callbacks)

    run.finish()
