import math
from os.path import dirname, realpath, join

import chess
import pylru
from tensorflow import keras
import numpy as np

from dinora.board_representation import canon_input_planes
from dinora.policy import move_lookup, flipped_move_lookup
from dinora.models.base import BaseModel, Priors, StateValue


MODELS_DIR = join(dirname(realpath(__file__)), "../../..", "models")
BEST_MODEL = join(MODELS_DIR, "latest.h5")


class DNNModel(BaseModel):
    def __init__(self, softmax_temp: float, model_path=BEST_MODEL) -> None:
        self.softmax_temp: float = softmax_temp
        self.model: keras.Model = keras.models.load_model(model_path)

    def model_out(self, board: chess.Board):
        flip = not board.turn
        plane = canon_input_planes(board.fen(), flip)
        plane = np.array([plane])
        model_output = self.model(plane, training=False)
        return model_output

    def raw_eval(self, board: chess.Board):
        policy, value = self.model_out(board)
        # unwrap tensor
        policy = policy[0]
        value = float(value[0])

        # take only legal moves from policy
        t = self.softmax_temp
        moves = []
        policies = []
        lookup = move_lookup if board.turn else flipped_move_lookup
        for move in board.legal_moves:
            i = lookup[move]
            moves.append(move)
            policies.append(math.exp(float(policy[i]) / t))

        # map to sum(policies) == 1
        s = sum(policies)
        policies_map = map(lambda e: math.exp(e / t) / s, policies)

        return dict(zip(moves, policies_map)), value

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        result = board.result(claim_draw=True)
        if result == "*":
            # Game is not ended
            # evaluate by using ANN
            priors, value_estimate = self.raw_eval(board)
        elif result == "1/2-1/2":
            # It's already draw
            # or we can claim draw, anyway `value_estimate` is 0.0
            # TODO: should I set priors = {}?
            # It's logical to set it empty because there is no need
            # to calculate deeper already draw position,
            # but with low time/nodes search, it leads to
            # empty node.children bug
            priors, _ = self.raw_eval(board)
            value_estimate = 0.0
        else:
            # result == '1-0' or result == '0-1'
            # we are checkmated because it's our turn to move
            # so the `value_estimate` is -1.0
            priors = {}  # no moves after checkmate
            value_estimate = -1.0
        return priors, value_estimate


class ChessModelWithCache:
    def __init__(self, size=200000, model_path=BEST_MODEL):
        self.cache = pylru.lrucache(size)
        self.net = DNNModel(model_path=model_path)

    def evaluate(self, board: chess.Board, softmax_temp):
        epd = board.epd()
        if epd in self.cache:
            policy, value = self.cache[epd]
            return policy, value
        else:
            policy, value = self.net.evaluate(board, softmax_temp)
            self.cache[epd] = [policy, value]
            return policy, value
