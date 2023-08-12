import chess
from dinora.board_representation import board_to_tensor


FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "1r3rk1/p1R2ppp/4p3/3pP3/3P1PP1/q1N1Q3/n6P/5RK1 w - - 0 21",
    "8/5kp1/8/4P1PP/4K3/8/5r2/8 b - - 2 60",
    "r1b1r1k1/pp1n1p2/2p5/3p2p1/4PpPp/2PP1P1P/PP4B1/R3KN1R w KQ - 0 18",
    "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    "rnbqkbnr/1pp1pp1p/p7/3pP3/6pP/3P4/PPP1BPP1/RNBQK1NR b KQkq h3 0 5",
]

BOARDS = [chess.Board(fen) for fen in FENS]


def bulk_boards(boards: list[chess.Board]):
    for board in boards:
        board_to_tensor(board)


def test_board_repr_new(benchmark):
    benchmark(bulk_boards, BOARDS)
