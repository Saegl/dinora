import math
import chess

from dinora.models.base import BaseModel, Priors, StateValue


def evaluate_board(board: chess.Board) -> float:
    mobility = board.legal_moves.count()
    return mobility / 40


def piece_value(piece: chess.Piece | None) -> float:
    if not piece:
        # Empty square
        return 0.0
    elif piece.piece_type == chess.PAWN:
        return 1.0
    elif piece.piece_type == chess.KNIGHT:
        return 3.0
    elif piece.piece_type == chess.BISHOP:
        return 3.2
    elif piece.piece_type == chess.ROOK:
        return 5.0
    elif piece.piece_type == chess.QUEEN:
        return 9.0
    elif piece.piece_type == chess.KING:
        return 90.0
    else:
        raise ValueError(f"Unsupported chess piece: {piece}")


def move_ordering(board: chess.Board):
    moves = []
    diffs = []
    for move in board.legal_moves:
        from_piece = board.piece_at(move.from_square)
        to_piece = board.piece_at(move.to_square)

        from_value = piece_value(from_piece)
        to_value = piece_value(to_piece)

        material_diff = to_value - from_value
        moves.append(move.uci())
        diffs.append(material_diff)

    bottom = sum(math.exp(x) for x in diffs)
    softmax = (math.exp(x) / bottom for x in diffs)

    return dict(zip(moves, softmax))


class DummyModel(BaseModel):
    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        return move_ordering(board), evaluate_board(board)
