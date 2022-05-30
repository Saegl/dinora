import logging

import chess
import chess.pgn

from .policy import policy_from_move, flip_policy
from .board_representation.canon_planes import canon_input_planes

logging.basicConfig(level=logging.DEBUG)


def num_result(game):
    result = game.headers["Result"]
    if result == "0-1":
        result = -1.0
    elif result == "1-0":
        result = 1.0
    elif result == "1/2-1/2":
        result = 0.0
    else:
        raise ValueError(f"Illegal game result: {result}")
    return result


def planes_input(fen: str, flip: bool):
    plane = canon_input_planes(fen, flip)
    return plane


def result_output(result: float, flip: bool):
    return -result if flip else result


def policy_output(move: chess.Move, flip: bool):
    policy = policy_from_move(move)
    if flip:
        policy = flip_policy(policy)
    return policy


def load_chess_games(filename_pgn: str, max_games: int):
    pgn = open(filename_pgn, "r", encoding="utf8", errors="ignore")
    game = True
    i = 0
    while game:
        game = chess.pgn.read_game(pgn)
        if not game or game.headers.get("Variant", "Standard") != "Standard":
            game = chess.pgn.read_game(pgn)
            continue
        i += 1
        if i > max_games:
            break
        if i % 1000 == 0:
            print(i)
        yield game


def chess_positions(games):
    for game in games:
        try:
            result = num_result(game)
        except ValueError:
            continue
        board = game.board()

        for move in game.mainline_moves():
            fen = board.fen()
            flip = not board.turn
            yield (
                planes_input(fen, flip),
                (policy_output(move, flip), result_output(result, flip)),
            )

            try:
                board.push(move)
            except AssertionError:
                logging.warning(
                    "Broken game found," f"can't make a move {move}." "Skipping"
                )
