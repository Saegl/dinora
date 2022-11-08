import sys
import traceback

import chess

from dinora.mcts import (
    run_mcts,
    MCTSparams,
    Constraint,
    TimeConstraint,
    NodesCountConstraint,
)


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
            from dinora.utils import disable_tensorflow_log

            disable_tensorflow_log()
            send("info string loading nn, it make take a while")
            import dinora.net

            self.net = dinora.net.ChessModel(softmax_temp)
            send("info string nn is loaded")

    def dispatcher(self, line: str):
        tokens = line.split()
        if tokens[0] == "uci":
            self.uci()
        elif tokens[0] == "isready":
            self.isready()
        elif tokens[0] == "ucinewgame":
            pass
        elif tokens[0] == "position":
            self.position(tokens)
        elif tokens[0] == "go":
            self.go(tokens)
        elif tokens[0] == "quit":
            sys.exit(0)
        else:
            send(f"info string command is not processed: {line}")

    def loop(self):
        while True:
            line = input()
            self.dispatcher(line)

    ### UCI Commands:
    def uci(self):
        send("id name Dinora Docker")
        send("id author Saegl")
        send("option name model type string default models/latest.h5")
        send("uciok")

    def isready(self):
        send("readyok")

    def position(self, tokens: list[str]):
        if tokens[1] == "startpos":
            self.board = chess.Board()
        if tokens[1] == "fen":
            fen = " ".join(tokens[2:8])
            self.board = chess.Board(fen)

        if "moves" in tokens:
            index = tokens.index("moves")
            moves = tokens[index + 1 :]
            for move_token in moves:
                self.board.push_uci(move_token)

    def go(self, tokens: list[str]):
        self.load_neural_network()
        constraint: Constraint
        # go wtime <wtime> btime <btime>
        if len(tokens) == 5 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            engine_time = wtime if self.board.turn else btime
            constraint = TimeConstraint(
                moves_number=self.board.fullmove_number,
                engine_time=engine_time,
                engine_inc=0,
            )
        # go wtime <wtime> btime <btime> winc <winc> binc <binc>
        elif len(tokens) == 9 and tokens[1] == "wtime":
            wtime = int(tokens[2])
            btime = int(tokens[4])
            winc = int(tokens[6])
            binc = int(tokens[8])
            engine_time = wtime if self.board.turn else btime
            engine_inc = winc if self.board.turn else binc
            constraint = TimeConstraint(
                moves_number=self.board.fullmove_number,
                engine_time=engine_time,
                engine_inc=engine_inc,
            )
        else:
            constraint = NodesCountConstraint(300)

        root_node = run_mcts(
            board=self.board,
            constraint=constraint,
            evaluator=self.net,
            params=MCTSparams(),
        )
        move = root_node.get_most_visited_move()
        send(f"bestmove {move}")


def start_uci(printlogs: bool = False):
    try:
        uci_state = UciState()
        uci_state.loop()
    except SystemExit:
        pass
    except:
        with open("dinora.log", "wt", encoding="utf8") as logfile:
            exc_type, exc_value, exc_tb = sys.exc_info()
            logfile.write(
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            )
            logfile.write("\n")

        if printlogs:
            with open("dinora.log", "rt", encoding="utf8") as f:
                print(f.read())
