"""
Dinora chess engine uses Deep Neural Network
to give prior probability of each move.

In NN, prior probabilities represented by one-hot encoding.
For that reason we need to generate all possible chess moves.
This module does exactly this

Generated moves are then stored in ALL_UCI_MOVES variable in UCI format

UCI move format is
    start_position + end_position + optional_promotion_piece_type
    where start_position = (start letter) + (start number)
        end_position = (end letter) + (end number)
        optional_promotion_piece_type = 'q' | 'r' | 'b' | 'n' | ''

Also note, to reduce possible NN inputs and increase training speed,
NN see board only from white perspective.
So when we want to make NN inference for black perspective
-> we have to flip board over horizontal line 
(white becomes black and black becomes white),
after this flip, all prior probabilites from NN also comes flipped.

There is 2 functions for training:
    move_to_policy
    move_to_flipped_policy

And 2 function for playing:
    move_prior_from_policy
    move_prior_from_flipped_policy

4 public functions overall
"""
from typing import Iterable
from itertools import product, chain

import chess
import numpy as np
import numpy.typing as npt


# rank (letter, horizontal), file (number, vertical)
Position = tuple[int, int]

LETTERS = list("abcdefgh")
NUMBERS = list(map(str, range(1, 9)))


def every_square() -> Iterable[Position]:
    "Return indices of 64 squares (position tuples)"
    for i, j in product(range(8), range(8)):
        yield i, j


def addsq(a: Position, b: Position) -> Position:
    "Add two position tuples"
    return a[0] + b[0], a[1] + b[1]


def position_to_uci_string(t: Position) -> str:
    return LETTERS[t[0]] + NUMBERS[t[1]]


def valid_tuple(t: Position) -> bool:
    return 0 <= t[0] <= 7 and 0 <= t[1] <= 7


vertical_transitions = [(t, 0) for t in range(-8, 8)]
horizontal_transitions = [(0, t) for t in range(-8, 8)]
diagonal_asc_transitions = [(t, t) for t in range(-8, 8)]
diagonal_desc_transitions = [(t, -t) for t in range(-8, 8)]
knight_transitions = [
    (-2, -1),
    (-1, -2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, 2),
    (2, 1),
    (1, 2),
]


def generate_uci_moves() -> list[str]:
    """
    Generate all possible moves for NN policy in UCI format
    """
    all_moves = []

    # First 'for' cycle
    # Covers Knight, Rook, Bishop, Queen,
    # Pawn staight moves, takes, but not promotions
    #
    # In UCI we don't specify piece type to move
    # so e4d5 covers pawn take, queen and bishop move
    for start_tuple in every_square():
        start_position = position_to_uci_string(start_tuple)

        for transition in chain(
            vertical_transitions,
            horizontal_transitions,
            diagonal_asc_transitions,
            diagonal_desc_transitions,
            knight_transitions,
        ):
            end_tuple = addsq(start_tuple, transition)

            if start_tuple == end_tuple:
                # Staying on the same square is not a valid move
                continue

            if not valid_tuple(end_tuple):
                # We are not on the 8*8 board
                continue

            end_position = position_to_uci_string(end_tuple)
            uci_move = start_position + end_position
            all_moves.append(uci_move)

    # Second 'for' cycle
    # Covers Pawn promotions
    # Note:
    #    NN agent always play from white perspective
    #    (we mirror board otherwise)
    #    so there is no need to cover promotions from black perspective

    promote_to = ["q", "r", "b", "n"]

    # White can promote only from 7th rank, 6th if we count from 0
    for start_tuple in [(t, 6) for t in range(8)]:
        start_position = position_to_uci_string(start_tuple)

        for transition in [(-1, 1), (0, 1), (1, 1)]:
            end_tuple = addsq(start_tuple, transition)

            if not valid_tuple(end_tuple):
                # We are not on the 8*8 board
                continue

            end_position = position_to_uci_string(end_tuple)
            uci_move = start_position + end_position

            for ptype in promote_to:
                promotion_move = uci_move + ptype
                all_moves.append(promotion_move)

    return all_moves


def flip_moves(moves: list[str]) -> list[str]:
    def flip_move(move: str) -> str:
        """
        After horizontal flip of chess board,
        only vertical components of move affected (files)
        """

        def flip_file(file: str) -> str:
            to_num = int(file)
            flipped = 9 - to_num
            return str(flipped)

        return "".join([(flip_file(c) if c.isdigit() else c) for c in move])

    return [flip_move(move) for move in moves]


ALL_UCI_MOVES = generate_uci_moves()
ALL_UCI_MOVES_LEN = len(ALL_UCI_MOVES)
assert ALL_UCI_MOVES_LEN == 1880  # Looks like there is 1880 possible moves

FLIPPED_UCI_MOVES = flip_moves(ALL_UCI_MOVES)

MOVE_LOOKUP = {
    chess.Move.from_uci(move): i
    for move, i in zip(ALL_UCI_MOVES, range(ALL_UCI_MOVES_LEN))
}
FLIPPED_MOVE_LOOKUP = {
    chess.Move.from_uci(move): i
    for move, i in zip(FLIPPED_UCI_MOVES, range(ALL_UCI_MOVES_LEN))
}

# Precompute all possible one-hot encoded policies
ONE_HOT_ENCODING_EYE = np.eye(ALL_UCI_MOVES_LEN, ALL_UCI_MOVES_LEN, dtype=np.float32)


def move_to_policy(move: chess.Move) -> npt.NDArray[np.float32]:
    "Move from white perspective to one-hot encoded tensor"
    index = MOVE_LOOKUP[move]
    return ONE_HOT_ENCODING_EYE[index]


def move_to_flipped_policy(move: chess.Move) -> npt.NDArray[np.float32]:
    "Move from black perspective to one-hot encoded tensor"
    index = FLIPPED_MOVE_LOOKUP[move]
    return ONE_HOT_ENCODING_EYE[index]


def policy_tensor(move: chess.Move, flip: bool):
    return move_to_flipped_policy(move) if flip else move_to_policy(move)


def move_prior_from_policy(
    policy: npt.NDArray[np.float32],
    move: chess.Move,
) -> float:
    "White move prior probability from one-hot encoded tensor"
    index = MOVE_LOOKUP[move]
    return policy[index]


def move_prior_from_flipped_policy(
    policy: npt.NDArray[np.float32],
    move: chess.Move,
) -> float:
    "Black move prior probability from one-hot encoded tensor"
    index = FLIPPED_MOVE_LOOKUP[move]
    return policy[index]
