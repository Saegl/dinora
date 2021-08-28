import sys
import traceback

import chess
import dinora.net
from dinora import search

from math import cos


EXTRA_TIME = 0.05
logfile = open("dinora.log", "w")
LOG = True


def log(msg):
    if LOG:
        logfile.write(str(msg))
        logfile.write("\n")
        logfile.flush()


def send(s: str):
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


def time_manager(moves_number: int, time_left: int, inc: int = 0) -> float:
    moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)
    remaining_time = time_left / 1000 + moves_left * inc / 1000
    move_time = remaining_time / moves_left - EXTRA_TIME
    return move_time


def uci_command(cmd: str, board: chess.Board, net):
    tokens = cmd.split()
    if tokens[0] == "uci":
        send("id name Dinora")
        send("id author Saegl")
        send("uciok")
    elif tokens[0] == "isready":
        send("readyok")
    elif tokens[0] == "ucinewgame":
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
    elif tokens[0] == "go":
        # go wtime <wtime> btime <btime>
        if len(tokens) == 5 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            engine_time = wtime if board.turn else btime
            move_time = time_manager(board.fullmove_number, engine_time)
            move, _ = search.uct_time(board, net, 4.0, move_time, send)
        # go wtime <wtime> btime <btime> winc <winc> binc <binc>
        elif len(tokens) == 9 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            winc = int(tokens[6])
            binc = int(tokens[8])
            engine_time = wtime if board.turn else btime
            engine_inc = winc if board.turn else binc
            move_time = time_manager(board.fullmove_number, engine_time, engine_inc)
            move, _ = search.uct_time(board, net, 4.0, move_time, send)
        else:
            move, _ = search.uct_nodes(board, 300, net, 4.0, send=send)
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
    try:
        start_uci()
    except:
        exc_type, exc_value, exc_tb = sys.exc_info()
        log(traceback.format_exception(exc_type, exc_value, exc_tb))
    finally:
        logfile.close()
