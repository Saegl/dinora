import pathlib
import dataclasses

from typing import Any

import chess

from dinora.mcts import MCTSparams, run_mcts, Constraint
from dinora.mcts.node import Node
from dinora.models import model_selector, BaseModel


class Engine:
    def __init__(
        self,
        model_name: str = "alphanet",
        weights_path: pathlib.Path | None = None,
        device: str | None = None,
    ):
        self._model_name = model_name
        self._model: BaseModel | None = None
        self.mcts_params = MCTSparams()
        self.weights_path = weights_path
        self.device = device

    @property
    def model(self) -> BaseModel:
        if self._model:
            return self._model
        else:
            raise Exception("Model is not loaded")

    def loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is None:
            self._model = model_selector(
                self._model_name, self.weights_path, self.device
            )

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def set_config_param(self, name: str, value: Any) -> None:
        for field in dataclasses.fields(MCTSparams):
            if field.name == name:
                setattr(self.mcts_params, name, field.type(value))
                break
        else:
            raise ValueError("Field is not found")

    def get_best_node(self, board: chess.Board, constraint: Constraint) -> Node:
        self.load_model()
        root_node = run_mcts(
            state=board,
            constraint=constraint,
            evaluator=self.model,
            params=self.mcts_params,
        )
        return root_node.best()

    def get_best_move(self, board: chess.Board, constraint: Constraint) -> chess.Move:
        node = self.get_best_node(board, constraint)
        assert node.move
        return node.move


if __name__ == "__main__":
    engine = Engine("alphanet")
    engine.set_config_param("cpuct", 4.0)
