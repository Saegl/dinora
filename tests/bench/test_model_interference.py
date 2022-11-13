"""
pytest .\tests\bench\test_model_interference.py --benchmark-warmup=on --benchmark-warmup-iterations=5
"""
import chess
import pytest
import numpy as np
from dinora.models.dnn import DNNModel
from dinora.models.badgyal import BadgyalModel
from dinora.models.handcrafted import DummyModel
from dinora.models.dnn.keras_model import build_model, LightConfig, ModelConfig


@pytest.mark.parametrize(
    "model",
    [
        build_model(LightConfig),
        # build_model(ModelConfig),
    ],
)
def test_dnn_interf(benchmark, model):
    boardtensor = np.zeros((1, 18, 8, 8), np.float32)
    # model.predict(boardtensor, training=False)
    benchmark(model, boardtensor, training=False)


@pytest.mark.parametrize(
    "config",
    [
        LightConfig,
        # ModelConfig,
    ],
)
def test_dnn_wrapped(benchmark, config):
    board = chess.Board()
    model = DNNModel()
    model.model = build_model(config)
    benchmark(model.evaluate, board)


if __name__ == "__main__":
    model = build_model(LightConfig)
    print(model.summary())
