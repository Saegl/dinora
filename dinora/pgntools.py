import logging
from typing import Iterator, TextIO

import chess
import chess.pgn
from chess import Board, Move
from chess.pgn import Game
import numpy as np
import numpy.typing as npt

from dinora.encoders.board_representation import board_to_tensor, board_to_compact_state
from dinora.encoders.policy import policy_index
from dinora.encoders.outcome import wdl_index

logging.basicConfig(level=logging.DEBUG)

npf32 = npt.NDArray[np.float32]


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


def load_state_tensors(pgn: TextIO) -> Iterator[tuple[npf32, tuple[int, int]]]:
    for game, board, move in load_game_states(pgn):
        flip = not board.turn
        yield (
            board_to_tensor(board),
            (
                policy_index(move, flip),
                wdl_index(game, flip),
            ),
        )


def load_compact_state_tensors(
    pgn: TextIO,
) -> Iterator[tuple[npf32, tuple[int, int]]]:
    for game, board, move in load_game_states(pgn):
        flip = not board.turn
        yield (
            board_to_compact_state(board),
            (
                policy_index(move, flip),
                wdl_index(game, flip),
            ),
        )
