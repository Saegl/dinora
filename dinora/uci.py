import sys
import traceback

import chess

from dinora.utils import disable_tensorflow_log

disable_tensorflow_log()
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


class UciState:
    def __init__(self):
        self.net = None
        self.board = chess.Board()

    def load_neural_network(self):
        if self.net is None:
            send("info string loading nn, it make take a while")
            import dinora.net

            self.net = dinora.net.ChessModel("models/latest.h5")
            send("info string nn is loaded")


def time_manager(moves_number: int, time_left: int, inc: int = 0) -> float:
    moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)
    remaining_time = time_left / 1000 + moves_left * inc / 1000
    move_time = remaining_time / moves_left - extra_time
    return move_time


def uci_command(state: UciState, cmd: str):
    tokens = cmd.split()
    if tokens[0] == "uci":
        send("id name Dinora Docker")
        send("id author Saegl")
        send("option name model type string default models/latest.h5")
        send("uciok")
    elif tokens[0] == "isready":
        state.load_neural_network()
        send("readyok")
    elif tokens[0] == "ucinewgame":
        pass
    elif tokens[0] == "position":
        if tokens[1] == "startpos":
            state.board = chess.Board()
        if tokens[1] == "fen":
            fen = " ".join(tokens[2:8])
            state.board = chess.Board(fen)

        if "moves" in tokens:
            index = tokens.index("moves")
            moves = tokens[index + 1 :]
            for move in moves:
                state.board.push_uci(move)
    elif tokens[0] == "go":
        state.load_neural_network()
        # go wtime <wtime> btime <btime>
        if len(tokens) == 5 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            engine_time = wtime if state.board.turn else btime
            move_time = time_manager(state.board.fullmove_number, engine_time)
            move, _ = search.uct_time(
                state.board,
                state.net,
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
            engine_time = wtime if state.board.turn else btime
            engine_inc = winc if state.board.turn else binc
            move_time = time_manager(
                state.board.fullmove_number, engine_time, engine_inc
            )
            move, _ = search.uct_time(
                state.board,
                state.net,
                c_puct,
                move_time,
                send,
                dirichlet_alpha,
                noise_eps,
                softmax_temp,
            )
        else:
            move, _ = search.uct_nodes(
                state.board,
                300,
                state.net,
                c_puct,
                send,
                dirichlet_alpha,
                noise_eps,
                softmax_temp,
            )
        send(f"bestmove {move}")
    elif cmd == "quit":
        exit()
    else:
        send(f"info string command is not processed: {cmd}")


def start_uci():
    uci_state = UciState()

    while True:
        line = input()
        uci_command(uci_state, line)


if __name__ == "__main__":
    try:
        start_uci()
    except:
        logfile = open("dinora.log", "w")
        exc_type, exc_value, exc_tb = sys.exc_info()
        logfile.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        logfile.write("\n")
        logfile.flush()
        logfile.close()
