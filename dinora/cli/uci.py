import sys
import traceback
from typing import Iterator, Any
from dataclasses import fields

import chess

from dinora.mcts import (
    run_mcts,
    MCTSparams,
    Constraint,
    InfiniteConstraint,
    TimeConstraint,
    NodesCountConstraint,
)
from dinora.models.base import BaseModel
from dinora.cli.uci_parser import parse_go_params
from dinora.cli.uci_options import UciOptions


def send(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


class UciState:
    def __init__(self) -> None:
        self.model: BaseModel | None = None  # model initialized after first `go` call
        self.board = chess.Board()
        self.mcts_params = MCTSparams()
        self.option_types: dict[str, UciOptions] = {}

    def get_options(
        self,
    ) -> Iterator[tuple[str, str, Any]]:
        for field in fields(self.mcts_params):
            if field.metadata.get("uci_option_type"):
                option_name = field.name
                option_type: UciOptions = field.metadata["uci_option_type"]
                option_type_name = option_type.uci_type
                option_default = field.default
                self.option_types[option_name] = option_type
                yield option_name, option_type_name, option_default

    def load_neural_network(self) -> None:
        if self.model is None:
            from dinora.utils import disable_tensorflow_log

            disable_tensorflow_log()
            send("info string loading nn, it make take a while")
            from dinora.models.dnn import DNNModel
            from dinora.models.cached_model import CachedModel

            # from dinora.models.badgyal import BadgyalModel as DNNModel

            # self.model = DNNModel(softmax_temp=1.6)
            self.model = CachedModel(DNNModel(softmax_temp=2.0))
            send("info string nn is loaded")

    def dispatcher(self, line: str) -> None:
        tokens = line.strip().split()
        if tokens[0] == "uci":
            self.uci()
        elif tokens[0] == "setoption":
            self.setoption(tokens)
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

    def loop(self) -> None:
        while True:
            line = input()
            self.dispatcher(line)

    ### UCI Commands:
    def uci(self) -> None:
        send("id name Dinora")
        send("id author Saegl")
        for name, type_name, default in self.get_options():
            send(f"option name {name} type {type_name} default {default}")
        send("uciok")

    def setoption(self, tokens: list[str]) -> None:
        i = tokens.index("value")
        name = tokens[i - 1].lower()
        value = tokens[i + 1]
        option_type = self.option_types[name]
        converted_value = option_type.convert(value)
        setattr(self.mcts_params, name, converted_value)

    def isready(self) -> None:
        send("readyok")

    def position(self, tokens: list[str]) -> None:
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

    def go(self, tokens: list[str]) -> None:
        self.load_neural_network()
        assert self.model  # Model loaded and it's not None

        go_params = parse_go_params(tokens)
        send(f"info string parsed params {go_params}")

        constraint: Constraint
        if go_params.infinite:
            constraint = InfiniteConstraint()

        elif time := go_params.is_time(self.board.turn):
            engine_time, engine_inc = time
            constraint = TimeConstraint(
                moves_number=self.board.fullmove_number,
                engine_time=engine_time,
                engine_inc=engine_inc,
            )

        elif nodes := go_params.nodes:
            constraint = NodesCountConstraint(nodes)

        else:
            constraint = InfiniteConstraint()

        send(f"info string chosen constraint {constraint}")

        root_node = run_mcts(
            board=self.board,
            constraint=constraint,
            evaluator=self.model,
            params=self.mcts_params,
        )
        move = root_node.get_most_visited_node().move
        send(f"bestmove {move}")


def start_uci(printlogs: bool = True) -> None:
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
