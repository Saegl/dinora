import logging

import chess
import chess.pgn
import numpy as np

from dinora.board_representation2 import board_to_tensor
from dinora.policy2 import policy_tensor

logging.basicConfig(level=logging.DEBUG)


def num_result(game):
    result = game.headers["Result"]
    if result == "0-1":
        result = 0.0
    elif result == "1-0":
        result = 1.0
    elif result == "1/2-1/2":
        result = 0.5
    else:
        raise ValueError(f"Illegal game result: {result}")
    return result


def result_output(result: float, flip: bool):
    return 1.0 - result if flip else result


def load_chess_games(filename_pgn: str):
    pgn = open(filename_pgn, "r", encoding="utf8", errors="ignore")
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        if not game or game.headers.get("Variant", "Standard") != "Standard":
            game = chess.pgn.read_game(pgn)
            continue

        yield game


def chess_positions(games):
    for game in games:
        try:
            result = num_result(game)
        except ValueError:
            continue
        board = game.board()

        for move in game.mainline_moves():
            flip = not board.turn

            yield (
                board_to_tensor(board),
                (
                    policy_tensor(move, flip),
                    np.array([result_output(result, flip)], dtype=np.float32),
                ),
            )

            try:
                board.push(move)
            except AssertionError:
                logging.warning(
                    "Broken game found," f"can't make a move {move}." "Skipping"
                )
