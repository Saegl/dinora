import pathlib

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime

from dinora.encoders.board_representation import board_to_tensor
from dinora.encoders.policy import extract_prob_from_policy
from dinora.models import search_weights
from dinora.models.base import BaseModel, Evaluation
from dinora.models.nnwrapper import NNWrapper

npf32 = npt.NDArray[np.float32]
DEFAULT_WEIGHTS_FILENAME = "alphanet_classic.ckpt.onnx"


def softmax(x: npf32, tau: float = 1.0) -> npf32:
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()  # type: ignore


class OnnxModel(BaseModel):
    def __init__(self, weights: pathlib.Path | None = None, device: str | None = None):
        if device is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device '{device}' is not supported")

        if weights is None:
            weights = search_weights(DEFAULT_WEIGHTS_FILENAME)

        self.ort_session = onnxruntime.InferenceSession(weights, providers=providers)

    def raw_outputs(self, board: chess.Board) -> tuple[npf32, npf32]:
        """
        NN outputs as numpy arrays
        priors numpy of shape (1880,)
        state value of shape (1,)
        """
        board_tensor = board_to_tensor(board)
        raw_policy, raw_value = self.ort_session.run(
            None, {"input": board_tensor.reshape(1, 18, 8, 8)}
        )
        return raw_policy[0], raw_value[0]

    def nn_evaluate(self, board: chess.Board) -> Evaluation:
        get_prob = extract_prob_from_policy

        raw_policy, raw_value = self.raw_outputs(board)

        moves = list(board.legal_moves)
        move_logits = [get_prob(raw_policy, move, not board.turn) for move in moves]

        move_priors = softmax(np.array(move_logits))
        priors = dict(zip(moves, move_priors))

        return priors, float(raw_value[0])

    def evaluate(self, board: chess.Board) -> Evaluation:
        return self.nn_evaluate(board)

    def raw_outputs_batch(self, boards: list[chess.Board]) -> tuple[npf32, npf32]:
        nn_tensor = []
        for board in boards:
            board_tensor = board_to_tensor(board)
            nn_tensor.append(board_tensor)

        raw_policies, raw_values = self.ort_session.run(
            None, {"input": np.array(nn_tensor)}
        )

        return raw_policies, raw_values

    def nn_evaluate_batch(self, board: chess.Board) -> list[Evaluation]:
        pass

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        return super().evaluate_batch(boards)

    def reset(self) -> None:
        pass
