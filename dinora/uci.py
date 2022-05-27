import sys
import traceback

import chess
import dinora.net
from dinora import search

from math import cos


extra_time = 0.5
c_puct = 2.0
softmax_temp = 1.6
dirichlet_alpha = 0.3
noise_eps = 0.00


def send(s: str):
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


def time_manager(moves_number: int, time_left: int, inc: int = 0) -> float:
    moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)
    remaining_time = time_left / 1000 + moves_left * inc / 1000
    move_time = remaining_time / moves_left - extra_time
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
    elif tokens[0] == "position":
        if tokens[1] == "startpos":
            board = chess.Board()
        if tokens[1] == "fen":
            fen = " ".join(tokens[2:8])
            board = chess.Board(fen)

        if "moves" in tokens:
            index = tokens.index("moves")
            moves = tokens[index + 1 :]
            for move in moves:
                board.push_uci(move)
    elif tokens[0] == "go":
        # go wtime <wtime> btime <btime>
        if len(tokens) == 5 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            engine_time = wtime if board.turn else btime
            move_time = time_manager(board.fullmove_number, engine_time)
            move, _ = search.uct_time(
                board,
                net,
                c_puct,
                move_time,
                send,
                dirichlet_alpha,
                noise_eps,
                softmax_temp,
            )
        # go wtime <wtime> btime <btime> winc <winc> binc <binc>
        elif len(tokens) == 9 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            winc = int(tokens[6])
            binc = int(tokens[8])
            engine_time = wtime if board.turn else btime
            engine_inc = winc if board.turn else binc
            move_time = time_manager(board.fullmove_number, engine_time, engine_inc)
            move, _ = search.uct_time(
                board,
                net,
                c_puct,
                move_time,
                send,
                dirichlet_alpha,
                noise_eps,
                softmax_temp,
            )
        else:
            move, _ = search.uct_nodes(
                board, 300, net, c_puct, send, dirichlet_alpha, noise_eps, softmax_temp
            )
        send(f"bestmove {move}")
    elif cmd == "quit":
        exit()
    else:
        send(f"info string command is not processed: {cmd}")

    return board, net


def start_uci():
    board = chess.Board()
    net = dinora.net.ChessModel("models/latest.h5")
    while True:
        board, net = uci_command(input(), board, net)


if __name__ == "__main__":
    try:
        start_uci()
    except:
        logfile = open("dinora.log", "w")
        exc_type, exc_value, exc_tb = sys.exc_info()
        logfile.write(traceback.format_exception(exc_type, exc_value, exc_tb))
        logfile.write("\n")
        logfile.flush()
        logfile.close()
