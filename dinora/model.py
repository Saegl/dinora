import chess
from tensorflow import keras
import numpy as np

from .board_utils import canon_input_planes
from .policy import move_lookup, flipped_move_lookup


class ChessModel:
    def __init__(self, model_path='data/mymodel2.h5'):
        self.model: keras.Model = keras.models.load_model(model_path)

    def raw_eval(self, board: chess.Board):
        plane = canon_input_planes(board.fen())
        plane = np.array([plane])
        model_output = self.model(plane, training=False)
        return model_output

    def eval(self, board: chess.Board):
        policy, value = self.raw_eval(board)
        # unwrap tensor
        policy = policy[0]
        value = float(value[0])

        # take only legal moves from policy
        moves = []
        policies = []
        lookup = move_lookup if board.turn else flipped_move_lookup
        for move in board.legal_moves:
            i = lookup[move]
            moves.append(move.uci())
            policies.append(float(policy[i]))

        # map to sum(policies) == 1
        s = sum(policies)
        policies = map(lambda e: e / s, policies)

        return dict(zip(moves, policies)), value


if __name__ == '__main__':
    with open('games.json', 'r', encoding='utf8') as f:
        import json
        buffer = json.load(f)
    buffer = buffer[0:800]
    net = ChessModel()

    import time
    start = time.time()
    for pos in buffer:
        board = chess.Board(
            fen=pos['fen'])
        policy, value = net.eval(board)
    print("800 fen = ", time.time() - start)

    priors = list(policy.items())
    priors.sort(key=lambda e: e[1])
    print(priors)
