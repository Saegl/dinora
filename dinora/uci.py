import sys
import chess
import dinora.net
from dinora import search


def send(s: str):
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


def uci_command(cmd: str, board: chess.Board, net):
    if cmd == "uci":
        send("id name Dinora")
        send("id author Saegl")
        send("uciok")
    elif cmd == "isready":
        send("readyok")
    elif cmd == "ucinewgame":
        pass
    elif cmd.startswith("position fen"):
        moves = []
        if "moves" in cmd:
            cmd, moves = cmd.split("moves")
            moves = moves.split(" ")[1:]
        fen = cmd.split(" ")[2:]
        fen = " ".join(fen)
        board = chess.Board(fen)
        for move in moves:
            board.push(chess.Move.from_uci(move))
    elif cmd.startswith("position startpos moves"):
        moves = cmd.split(" ")[3:]
        board = chess.Board()
        for move in moves:
            board.push(chess.Move.from_uci(move))
    elif cmd.startswith("go"):
        move, score = search.uct(board, 300, net, 4.0, send=send)
        send(f"bestmove {move}")
    elif cmd == "quit":
        exit()
    else:
        send(f"info string command is not processed: {cmd}")

    return board, net


def start_uci():
    board = chess.Board()
    net = dinora.net.ChessModel()
    while True:
        board, net = uci_command(input(), board, net)


if __name__ == "__main__":
    start_uci()
