import sys
from typing import Iterator, Any
from dataclasses import fields

import chess

from dinora.mcts import (
    run_mcts,
    Node,
    MCTSparams,
    Constraint,
    InfiniteConstraint,
    TimeConstraint,
    MoveTimeConstraint,
    NodesCountConstraint,
)
from dinora.models import BaseModel, model_selector
from dinora.uci.uci_parser import parse_go_params
from dinora.uci.uci_options import UciOptions


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
        self.tree: Node | None = None

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

    def load_model(self) -> None:
        if self.model is None:
            send("info string loading nn, it make take a while")
            self.model = model_selector("alphanet")  # TODO dont hardcode
            send("info string nn is loaded")

    def dispatcher(self, line: str) -> None:
        tokens = line.strip().split()
        if tokens[0] == "uci":
            self.uci()
        elif tokens[0] == "ucinewgame":
            self.ucinewgame()
        elif tokens[0] == "setoption":
            self.setoption(tokens)
        elif tokens[0] == "isready":
            self.isready()
        elif tokens[0] == "position":
            self.position(tokens)
        elif tokens[0] == "go":
            self.go(tokens)
        elif tokens[0] == "quit":
            sys.exit(0)
        else:
            send(f"info string command is not processed: {line}")

    def loop(self) -> None:
        send("Dinora chess engine")
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

    def ucinewgame(self) -> None:
        self.tree = None
        self.load_model()
        self.board = chess.Board()

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

                if self.tree:
                    for node in self.tree.children.values():
                        if node.board.epd() == self.board.epd():
                            self.tree = node

    def go(self, tokens: list[str]) -> None:
        self.load_model()
        assert self.model  # Model loaded and it's not None

        go_params = parse_go_params(tokens)
        send(f"info string parsed params {go_params}")

        constraint: Constraint
        if go_params.infinite:
            constraint = InfiniteConstraint()

        elif time := go_params.movetime:
            constraint = MoveTimeConstraint(go_params.movetime)

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

        if self.tree and self.board.epd() == self.tree.board.epd():
            tree = self.tree
        else:
            tree = None

        root_node = run_mcts(
            state=tree if tree else self.board,
            constraint=constraint,
            evaluator=self.model,
            params=self.mcts_params,
        )
        self.tree = root_node
        move = root_node.get_most_visited_node().move  # Robust and non robust move?
        send(f"bestmove {move}")


def start_uci() -> None:
    uci_state = UciState()
    uci_state.loop()