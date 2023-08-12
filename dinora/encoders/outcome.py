import chess.pgn
import chess.engine
from chess import WHITE, BLACK

WIN, DRAW, LOSS = range(3)


class UnexpectedOutcome(Exception):
    pass


def wdl_index(game: chess.pgn.Game, turn: bool):
    result = game.headers["Result"]

    if result == "1/2-1/2":
        return DRAW
    elif turn == WHITE and result == "1-0":
        return WIN
    elif turn == WHITE and result == "0-1":
        return LOSS
    elif turn == BLACK and result == "1-0":
        return LOSS
    elif turn == BLACK and result == "0-1":
        return WIN
    else:
        raise UnexpectedOutcome(f"Illegal game result: {result}")


def z_value(game: chess.pgn.Game, turn: bool):
    result = game.headers["Result"]

    if result == "1/2-1/2":
        return 0.0
    elif turn == WHITE and result == "1-0":
        return +1.0
    elif turn == WHITE and result == "0-1":
        return -1.0
    elif turn == BLACK and result == "1-0":
        return -1.0
    elif turn == BLACK and result == "0-1":
        return +1.0
    else:
        raise UnexpectedOutcome(f"Illegal game result: {result}")


def stockfish_value(board: chess.Board, engine: chess.engine.SimpleEngine, nodes: int):
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))

    # [0:1] scale
    wdl = info["wdl"].pov(board.turn).expectation()

    # [-1:1] scale
    wdl_scaled = wdl * 2.0 - 1.0
    return wdl_scaled
