import logging
from typing import Iterator, TextIO

import chess
import chess.pgn
from chess import Board, Move
from chess.pgn import Game
import numpy as np
import numpy.typing as npt

from dinora.board_representation2 import board_to_tensor, board_to_compact_state
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


def load_chess_games(pgn: TextIO) -> Iterator[chess.pgn.Game]:
    def is_supported_variant(game):
        return game.headers.get("Variant", "Standard") == "Standard"

    game = chess.pgn.read_game(pgn)
    while game:
        if is_supported_variant(game):
            yield game
        
        game = chess.pgn.read_game(pgn)


def load_game_states(pgn: TextIO) -> Iterator[tuple[Game, Board, Move]]:
    for game in load_chess_games(pgn):
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


def load_state_tensors(pgn: TextIO) -> Iterator[tuple[npf32, tuple[npf32, npf32]]]:
    for game, board, move in load_game_states(pgn):
        flip = not board.turn
        yield (
            board_to_tensor(board),
            (
                policy_index_tensor(move, flip),
                outcome_tensor(game, flip),
            ),
        )


def load_compact_state_tensors(pgn: TextIO) -> Iterator[tuple[npf32, tuple[npf32, npf32]]]:
    for game, board, move in load_game_states(pgn):
        flip = not board.turn
        yield (
            board_to_compact_state(board),
            (
                policy_index_tensor(move, flip),
                outcome_tensor(game, flip),
            ),
        )
