import sys
import traceback

import chess

from dinora.utils import disable_tensorflow_log

disable_tensorflow_log()
from dinora.mcts.search import run_mcts, MCTSparams
from dinora.mcts.constraints import TimeConstraint, NodesCountConstraint


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

            self.net = dinora.net.ChessModel(softmax_temp)
            send("info string nn is loaded")


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
            for move_token in moves:
                state.board.push_uci(move_token)
    elif tokens[0] == "go":
        state.load_neural_network()
        # go wtime <wtime> btime <btime>
        if len(tokens) == 5 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            engine_time = wtime if state.board.turn else btime
            constraint = TimeConstraint(
                moves_number=state.board.fullmove_number,
                engine_time=engine_time,
                engine_inc=0,
            )
        # go wtime <wtime> btime <btime> winc <winc> binc <binc>
        elif len(tokens) == 9 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            winc = int(tokens[6])
            binc = int(tokens[8])
            engine_time = wtime if state.board.turn else btime
            engine_inc = winc if state.board.turn else binc
            constraint = TimeConstraint(
                moves_number=state.board.fullmove_number,
                engine_time=engine_time,
                engine_inc=engine_inc,
            )
        else:
            constraint = NodesCountConstraint(300)

        root_node = run_mcts(
            board=state.board,
            constraint=constraint,
            evaluator=state.net,
            params=MCTSparams(),
        )
        move = root_node.get_most_visited_move()
        send(f"bestmove {move}")
    elif cmd == "quit":
        sys.exit()
    else:
        send(f"info string command is not processed: {cmd}")


def start_uci():
    try:
        uci_state = UciState()

        while True:
            line = input()
            uci_command(uci_state, line)
    except:
        logfile = open("dinora.log", "w")
        exc_type, exc_value, exc_tb = sys.exc_info()
        logfile.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        logfile.write("\n")
        logfile.flush()
        logfile.close()
