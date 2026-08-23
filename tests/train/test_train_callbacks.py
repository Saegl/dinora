import pytest

from dinora.models.alphanet import AlphaNet
from train.train_callbacks import (
    BoardsEvaluator,
    SampleGameGenerator,
    ValidationCheckpointer,
)
from train.trainer import Trainer

micro_alphanet_conf = {
    "filters": 16,
    "res_blocks": 2,
    "policy_channels": 8,
    "value_channels": 2,
    "value_fc_hidden": 32,
    "learning_rate": 0.001,
    "optimizer_name": "Adam",
    "optimizer_params": {},
    "scheduler_name": "StepLR",
    "scheduler_params": {
        "step_size": 1,
        "gamma": 1.0,
    },
    "scheduler_frequency": 1000,
}


class MockLogger:
    @property
    def name(self) -> str:
        return "MockLogger"

    @property
    def version(self) -> int:
        return -1

    def log_metrics(self, metrics, step=None) -> None:
        pass

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        pass

    def log_text(self, *args, **kwargs) -> None:
        pass


def test_sample_game_generator():
    callback = SampleGameGenerator()
    trainer = Trainer(logger=MockLogger())
    pl_module = AlphaNet(**micro_alphanet_conf)

    callback.on_fit_start(trainer, pl_module)
    callback.on_validation_end(trainer, pl_module)


def test_boards_evaluator():
    callback = BoardsEvaluator()
    trainer = Trainer(logger=MockLogger())
    pl_module = AlphaNet(**micro_alphanet_conf)

    callback.on_validation_end(trainer, pl_module)


@pytest.mark.disk_usage
def test_validation_checkpointer():
    callback = ValidationCheckpointer()
    trainer = Trainer(logger=MockLogger())
    pl_module = AlphaNet(**micro_alphanet_conf)

    import wandb

    wandb.init(mode="disabled")

    callback.on_validation_end(trainer, pl_module)
    callback.on_train_end(trainer, pl_module)
