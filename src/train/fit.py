import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pprint import pprint
from typing import Any, Literal

import torch
import wandb

from dinora import PROJECT_ROOT
from train.callback import Callback
from train.datamodules import WandbDataModule
from train.elofish_callback import ElofishRatingEstimator
from train.train_callbacks import (
    BoardsEvaluator,
    CPLoss,
    SampleGameGenerator,
    TrainerCheckpointer,
    ValidationCheckpointer,
)
from train.trainer import Trainer
from train.tuner import Tuner
from train.wandb_logger import WandbLogger

logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)


WANDB_LOGS_DIR = pathlib.Path("logs/wandb_logs")


@dataclass(frozen=True)
class AlphaNetConfig:
    res_channels: int
    res_blocks: int
    policy_channels: int
    value_channels: int
    value_lin_channels: int


@dataclass(frozen=True)
class SeNetConfig(AlphaNetConfig):
    pass


@dataclass
class Config:
    matmul_precision: Literal["highest", "high", "medium"]
    max_time: dict | None  # type: ignore
    max_epochs: int  # set -1 to ignore
    dataset_label: str
    trainer_ckpt_label: str | None

    z_weight: float
    q_weight: float
    value_loss_weight: float

    tune_batch: bool
    batch_size: int  # will be overwritten if tune_batch = True

    tune_learning_rate: bool
    learning_rate: float  # will be overwritten if tune_learning_rate = True

    optimizer_name: str
    optimizer_params: dict[str, Any]

    scheduler_name: str
    scheduler_params: dict[str, Any]
    scheduler_frequency: int

    enable_checkpointing: bool
    checkpoint_train_time_interval: dict  # type: ignore

    enable_sample_game_generator: bool
    enable_boards_evaluator: bool
    enable_validation_checkpointer: bool
    enable_trainer_checkpointer: bool

    enable_cploss: bool
    cploss_label: str
    cploss_batch_size: int
    cploss_positions: int

    enable_elofish: bool
    elofish_games_count: int
    elofish_step_freq: int
    elofish_time_limit: float
    elofish_rating: float
    elofish_student_deviation: float
    elofish_student_command_prefix: list[str]
    elofish_teacher_command: list[str]
    elofish_upload_reports: bool

    log_every_n_steps: int

    val_check_interval: float | int

    limit_train_batches: int | None
    limit_val_batches: int | None
    limit_test_batches: int | None

    model_type: Literal["alphanet", "senet"]
    model_conf: AlphaNetConfig | SeNetConfig

    # TODO: add use example to jupyter notebook
    @staticmethod
    def from_file(path: pathlib.Path) -> "Config":
        with path.open("r") as f:
            config = Config.from_dict(json.load(f))
        return config

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Config":
        ModelConfig = AlphaNetConfig if d["model_type"] == "alphanet" else SeNetConfig
        model_conf = ModelConfig(**d["model_conf"])
        conf = Config(**(d | {"model_conf": model_conf}))
        return conf


class LearningRateMonitor(Callback):
    """Logs ``lr-<OptimizerClassName>``, suffixed ``/pg1`` etc. for multiple groups.

    A ``-momentum`` key carries ``betas[0]`` for Adam-like optimizers, ``momentum`` for SGD.
    """

    def _optimizer_stats(self, optimizer: Any) -> dict[str, float]:
        stats: dict[str, float] = {}
        name = "lr-" + type(optimizer).__name__
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults
        for i, group in enumerate(param_groups):
            pg_name = name if len(param_groups) == 1 else f"{name}/pg{i + 1}"
            stats[pg_name] = group["lr"]
            momentum = group["betas"][0] if use_betas else group.get("momentum", 0)
            stats[f"{pg_name}-momentum"] = momentum
        return stats

    def on_train_batch_start(
        self, trainer: Trainer, model: Any, batch: Any, batch_idx: int
    ) -> None:
        # global_step is not incremented until after this callback
        next_step = trainer.global_step + 1
        if next_step % trainer.log_every_n_steps == 0 and trainer.logger:
            stats: dict[str, float] = {}
            for optimizer in trainer.optimizers:
                stats.update(self._optimizer_stats(optimizer))
            if stats:
                trainer.logger.log_metrics(stats, step=next_step)


class ModelSummary(Callback):
    def on_fit_start(self, trainer: Trainer, model: Any) -> None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {type(model).__name__}")
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}\n")


class ModelCheckpoint(Callback):
    def __init__(
        self,
        dirpath: pathlib.Path,
        filename: str,
        save_weights_only: bool,
        train_time_interval: timedelta,
    ) -> None:
        self.dirpath = pathlib.Path(dirpath)
        self.filename = filename
        self.save_weights_only = save_weights_only
        self.train_time_interval = train_time_interval
        self._last_save_time: float | None = None

    def on_fit_start(self, trainer: Trainer, model: Any) -> None:
        self._last_save_time = time.time()

    def on_train_batch_start(
        self, trainer: Trainer, model: Any, batch: Any, batch_idx: int
    ) -> None:
        if self._last_save_time is None:
            return
        elapsed = time.time() - self._last_save_time
        if elapsed >= self.train_time_interval.total_seconds():
            self._save(trainer, model)
            self._last_save_time = time.time()

    def _save(self, trainer: Trainer, model: Any) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        filename = self.filename.format(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
        )
        filepath = self.dirpath / f"{filename}.ckpt"
        if self.save_weights_only:
            torch.save(model.state_dict(), filepath)
        else:
            torch.save(model, filepath)
        print(f"Checkpoint saved: {filepath}")

        after_save = getattr(trainer.logger, "after_save_checkpoint", None)
        if after_save is not None:
            after_save(
                filepath,
                {
                    "epoch": trainer.current_epoch,
                    "step": trainer.global_step,
                    "original_filename": filepath.name,
                    "save_weights_only": self.save_weights_only,
                },
            )


def get_model(config: Config) -> Any:
    if config.model_type == "alphanet":
        from dinora.models.alphanet import AlphaNet

        return AlphaNet(
            filters=config.model_conf.res_channels,
            res_blocks=config.model_conf.res_blocks,
            policy_channels=config.model_conf.policy_channels,
            value_channels=config.model_conf.value_channels,
            value_fc_hidden=config.model_conf.value_lin_channels,
            value_loss_weight=config.value_loss_weight,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer_name,
            optimizer_params=config.optimizer_params,
            scheduler_name=config.scheduler_name,
            scheduler_params=config.scheduler_params,
            scheduler_frequency=config.scheduler_frequency,
        )
    elif config.model_type == "senet":
        from dinora.models.senet import SeNet

        return SeNet(
            filters=config.model_conf.res_channels,
            res_blocks=config.model_conf.res_blocks,
            policy_channels=config.model_conf.policy_channels,
            value_channels=config.model_conf.value_channels,
            value_fc_hidden=config.model_conf.value_lin_channels,
            value_loss_weight=config.value_loss_weight,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer_name,
            optimizer_params=config.optimizer_params,
            scheduler_name=config.scheduler_name,
            scheduler_params=config.scheduler_params,
            scheduler_frequency=config.scheduler_frequency,
        )
    else:
        raise ValueError("This model is not supported")


def fit(config: Config) -> None:  # noqa: C901
    run = wandb.init(project="dinora-chess", dir=WANDB_LOGS_DIR)
    run.config.update({"config_file": asdict(config)})
    pprint(config)

    wandb_logger = WandbLogger(
        project="dinora-chess",
        log_model="all",  # save model weights to wandb
    )

    torch.set_float32_matmul_precision(config.matmul_precision)
    max_time = timedelta(**config.max_time) if config.max_time else None

    callbacks: list[Callback] = [
        LearningRateMonitor(),
        ModelSummary(),
    ]

    if config.enable_sample_game_generator:
        callbacks.append(SampleGameGenerator())

    if config.enable_boards_evaluator:
        callbacks.append(BoardsEvaluator())

    if config.enable_validation_checkpointer:
        callbacks.append(ValidationCheckpointer())

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

    if config.enable_trainer_checkpointer:
        callbacks.append(TrainerCheckpointer())

    if config.enable_cploss:
        callbacks.append(
            CPLoss(
                config.cploss_label,
                config.cploss_positions,
                config.cploss_batch_size,
            )
        )

    if config.enable_elofish:
        callbacks.append(
            ElofishRatingEstimator(
                config.elofish_games_count,
                config.elofish_step_freq,
                config.elofish_time_limit,
                config.elofish_rating,
                config.elofish_student_deviation,
                config.elofish_student_command_prefix,
                config.elofish_teacher_command,
                config.elofish_upload_reports,
            )
        )

    model = get_model(config)

    datamodule = WandbDataModule(
        dataset_label=config.dataset_label,
        batch_size=config.batch_size,
        z_weight=config.z_weight,
        q_weight=config.q_weight,
    )

    trainer = Trainer(
        max_time=max_time,
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        enable_checkpointing=config.enable_checkpointing,
        default_root_dir=PROJECT_ROOT / "checkpoints",
        callbacks=callbacks,
        val_check_interval=config.val_check_interval,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
    )

    tuner = Tuner(trainer)

    if config.tune_batch:
        config.batch_size = tuner.scale_batch_size(model, datamodule)

    if config.tune_learning_rate:
        suggested = tuner.lr_find(model, datamodule)
        if suggested is not None:
            config.learning_rate = suggested

    ckpt_path = None

    if config.trainer_ckpt_label:
        ckpt_dir = pathlib.Path()
        ckpt_path = ckpt_dir / "trainer.ckpt"

        file = run.use_artifact(config.trainer_ckpt_label)
        file.download(root=ckpt_dir)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
    run.finish()


def validate(config: Config) -> None:
    model = torch.load("models/model-eliteq.ckpt")

    wandb.init(project="dinora-chess", dir=WANDB_LOGS_DIR)

    datamodule = WandbDataModule(
        dataset_label=config.dataset_label,
        batch_size=config.batch_size,
        z_weight=config.z_weight,
        q_weight=config.q_weight,
    )

    trainer = Trainer(
        limit_val_batches=config.limit_val_batches,
    )
    trainer.validate(model, datamodule)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Provide path to config, examples at configs/train/dev.json",
        type=pathlib.Path,
    )

    args = parser.parse_args()
    config_path = args.config

    config = Config.from_file(config_path)

    fit(config)
