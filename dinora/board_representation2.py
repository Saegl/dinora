"""
The only function here is `board_to_tensor`.
It takes chess.Board and returns a numpy array of shape (18, 8, 8) 
for keras CNN input in NCHW format.

NCHW format:
    N is the number of boards = 1
    C [channels] is the number of planes
        (12 for pieces, 4 for castling, 1 for fifty move, 1 for en passant)
        = 12 + 4 + 1 + 1 = 18
    H [height] of chess board = 8
    W [width] of chess board = 8

Array weight is 18 * 8 * 8 * 4 = 4608 bytes, where 4 is float32
"""
import chess
import numpy as np
import numpy.typing as npt


plane_index = {
    (chess.KING, chess.WHITE): 0,
    (chess.QUEEN, chess.WHITE): 1,
    (chess.ROOK, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.KNIGHT, chess.WHITE): 4,
    (chess.PAWN, chess.WHITE): 5,
    (chess.KING, chess.BLACK): 6,
    (chess.QUEEN, chess.BLACK): 7,
    (chess.ROOK, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.KNIGHT, chess.BLACK): 10,
    (chess.PAWN, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> npt.NDArray[np.float32]:
    "Convert current state (chessboard) to tensor"
    flip = not board.turn
    if flip:
        board = board.mirror()

    tensor = np.zeros((18, 8, 8), np.float32)

    # Set pieces [0: 12)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            index = plane_index[piece.piece_type, piece.color]
            tensor[index][square // 8][(square % 8)] = 1.0

    # Set castling rights [12: 16)
    if board.castling_rights & chess.BB_H1:
        tensor[12].fill(1.0)
    if board.castling_rights & chess.BB_A1:
        tensor[13].fill(1.0)
    if board.castling_rights & chess.BB_H8:
        tensor[14].fill(1.0)
    if board.castling_rights & chess.BB_A8:
        tensor[15].fill(1.0)

    # Set fifty move counter [16: 17)
    tensor[16].fill(float(board.halfmove_clock))

    # Set en passant square [17: 18)
    if board.has_legal_en_passant():
        square: chess.Square = board.ep_square
        tensor[17][square // 8][square % 8] = 1.0

    return tensor
