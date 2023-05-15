import logging
from typing import Iterator

import chess
import chess.pgn
from chess import Board, Move
from chess.pgn import Game
import numpy as np
import numpy.typing as npt

from dinora.board_representation2 import board_to_tensor
from dinora.policy2 import policy_index_tensor

logging.basicConfig(level=logging.DEBUG)

npf32 = npt.NDArray[np.float32]

WHITE_WON = np.array(0, dtype=np.int64)
DRAW = np.array(1, dtype=np.int64)
BLACK_WON = np.array(2, dtype=np.int64)

class UnexpectedOutcome(Exception):
    pass


def outcome_tensor(game, flip: bool):
    result = game.headers["Result"]

    if result == "0-1":
        return BLACK_WON if not flip else WHITE_WON
    elif result == "1-0":
        return WHITE_WON if not flip else BLACK_WON
    elif result == "1/2-1/2":
        return DRAW
    else:
        raise UnexpectedOutcome(f"Illegal game result: {result}")


def load_chess_games(filename_pgn: str) -> Iterator[chess.pgn.Game]:
    with open(filename_pgn, "r", encoding="utf8", errors="ignore") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if not game or game.headers.get("Variant", "Standard") != "Standard":
                game = chess.pgn.read_game(pgn)
                continue

            yield game


def load_game_states(filename_pgn: str) -> Iterator[tuple[Game, Board, Move]]:
    for game in load_chess_games(filename_pgn):
        board = game.board()
        for move in game.mainline_moves():
            yield game, board, move
            
            try:
                board.push(move)
            except AssertionError:
                logging.warning(
                    "Broken game found," f"can't make a move {move}." "Skipping game"
                )
                break


def load_state_tensors(filename_pgn: str) -> Iterator[tuple[npf32, tuple[npf32, npf32]]]:
    for game, board, move in load_game_states(filename_pgn):
        flip = not board.turn
        try:
            yield (
                board_to_tensor(board),
                (
                    policy_index_tensor(move, flip),
                    outcome_tensor(game, flip),
                ),
            )
        except UnexpectedOutcome as e:
            logging.warning(f"Cannot encode game outcome: {e}")
            break
