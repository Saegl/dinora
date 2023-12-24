import chess
import numpy as np
import onnxruntime

from dinora.models.base import BaseModel, Priors, StateValue
from dinora.encoders.board_representation import board_to_tensor
from dinora.encoders.policy import extract_prob_from_policy


def softmax(x, tau=1.0):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


class OnnxModel(BaseModel):
    def __init__(self, weights, device: str):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device '{device}' is not supported")

        self.ort_session = onnxruntime.InferenceSession(weights, providers=providers)

    def raw_evaluate(self, board: chess.Board):
        board_tensor = board_to_tensor(board)
        get_prob = extract_prob_from_policy

        raw_policy, raw_value = self.ort_session.run(
            None, {"input": board_tensor.reshape(1, 18, 8, 8)}
        )

        outcome_logits = float(raw_value[0])

        policy = raw_policy[0]

        moves = list(board.legal_moves)
        move_logits = [get_prob(policy, move, not board.turn) for move in moves]

        move_priors = softmax(np.array(move_logits))
        priors = dict(zip(moves, move_priors))

        return priors, outcome_logits

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        result = board.result(claim_draw=True)
        if result == "*":
            # Game is not ended
            # evaluate by using ANN
            priors, value_estimate = self.raw_evaluate(board)
        elif result == "1/2-1/2":
            # It's already draw
            # or we can claim draw, anyway `value_estimate` is 0.0
            # TODO: should I set priors = {}?
            # It's logical to set it empty because there is no need
            # to calculate deeper already draw position,
            # but with low time/nodes search, it leads to
            # empty node.children bug
            priors, _ = self.raw_evaluate(board)
            value_estimate = 0.0
        else:
            # result == '1-0' or result == '0-1'
            # we are checkmated because it's our turn to move
            # so the `value_estimate` is -1.0
            priors = {}  # no moves after checkmate
            value_estimate = -1.0
        return priors, value_estimate

    def reset(self):
        pass
