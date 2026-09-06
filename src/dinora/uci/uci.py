import contextlib
import queue
import threading
from dataclasses import dataclass
from typing import Any

import chess

import dinora
from dinora.engine import Engine, ParamNotFound
from dinora.options import uci_options
from dinora.search.stoppers import Stopper
from dinora.uci.output import send
from dinora.uci.uci_go_parser import parse_go_params

DEFAULT_MOVE_OVERHEAD_MS = 600


class UciCommand:
    """Command sent by UciLoop"""


@dataclass
class Go(UciCommand):
    board: chess.Board
    stopper: Stopper


@dataclass
class SetOption(UciCommand):
    name: str
    value: str


@dataclass
class Quit(UciCommand):
    pass


_load_lock = threading.Lock()


def load_model(engine: Engine) -> None:
    """
    Load the network, once, reporting progress over UCI.

    Called from both threads: the reader thread on `isready`, the engine thread
    on `go` in case the GUI skipped `isready`.
    """
    with _load_lock:
        if engine.loaded():
            return

        send(f"info string searcher <{engine.searcher.name()}>")
        send("info string model is loading")
        engine.load_model()
        send(f"info string model loaded type <{engine.model.name()}>")


def uci_start(engine: Engine) -> None:
    commands_queue: queue.Queue[UciCommand] = queue.Queue()
    uciloop = UciCommunicator(commands_queue, engine)
    uciengine = UciEngine(commands_queue, engine)

    controller_thread = threading.Thread(target=uciengine.loop)
    controller_thread.start()

    uciloop.start_communication()


@dataclass
class UciEngine:
    commands_queue: queue.Queue[UciCommand]
    engine: Engine

    def loop(self) -> None:
        running = True
        while running:
            command = self.commands_queue.get()
            match command:
                case Go(board, stopper):
                    load_model(self.engine)
                    move = self.engine.get_best_move(board, stopper)
                    stopper.early_stop.set()
                    send(f"bestmove {move}")
                case SetOption(name, value):
                    with contextlib.suppress(ParamNotFound):
                        self.engine.set_config_param(name, value)
                case Quit():
                    running = False

            self.commands_queue.task_done()
            if not running:
                break


class UciCommunicator:
    commands_queue: queue.Queue[UciCommand]
    engine: Engine
    params: Any
    active_stopper: Stopper | None
    board: chess.Board
    running: bool

    def __init__(self, commands_queue: queue.Queue[UciCommand], engine: Engine):
        self.commands_queue = commands_queue
        self.engine = engine
        self.params = engine.searcher.params
        self.active_stopper = None
        self.board = chess.Board()
        self.running = True

        self.option_move_overhead_ms = DEFAULT_MOVE_OVERHEAD_MS

    def start_communication(self) -> None:
        send(f"Dinora Chess Engine v{dinora.__version__}")
        while self.running:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                self.quit([])
                break

            if not line.strip():
                continue  # Ignore empty or whitespace-only lines

            self.dispatcher(line)

    def dispatcher(self, line: str) -> None:
        command, *tokens = line.strip().split()
        supported_commands = [
            "uci",
            "ucinewgame",
            "setoption",
            "isready",
            "position",
            "go",
            "stop",
            "quit",
        ]
        if command in supported_commands:
            command_method = getattr(self, command)
            try:
                command_method(tokens)
            except Exception as e:
                send(f"info string '{command}' failed with '{e}'")
        else:
            send(f"info string command is not processed: {line}")

    def uci(self, _: list[str]) -> None:
        send(f"id name Dinora v{dinora.__version__}")
        send("id author Saegl")

        for option in uci_options(self.params):
            send(option.line())

        send(
            f"option name move_overhead_ms type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max 10000"
        )
        send("uciok")

    def ucinewgame(self, _: list[str]) -> None:
        self.board = chess.Board()

    def setoption(self, tokens: list[str]) -> None:
        name = tokens[tokens.index("name") + 1].lower()
        value = tokens[tokens.index("value") + 1]

        if name == "move_overhead_ms":
            self.option_move_overhead_ms = int(value)

        self.commands_queue.put(SetOption(name, value))

    def isready(self, _: list[str]) -> None:
        """
        `readyok`, but only once the engine is done initializing.

        The load runs on this thread, off the command queue, so an `isready`
        during a search is answered at once instead of waiting for `bestmove`.
        """
        load_model(self.engine)
        send("readyok")

    def position(self, tokens: list[str]) -> None:
        if tokens[0] == "startpos":
            self.board = chess.Board()
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            self.board = chess.Board(fen)
        else:
            raise ValueError('Unexpected token after "position"')

        if "moves" in tokens:
            moves = tokens[tokens.index("moves") + 1 :]
            for move_token in moves:
                self.board.push_uci(move_token)

    def go(self, tokens: list[str]) -> None:
        if self.active_stopper and not self.active_stopper.early_stop.is_set():
            raise Exception("Can't run second `go`")

        go_params = parse_go_params(tokens)
        send(f"info string parsed params {go_params}")

        self.active_stopper = go_params.get_search_stopper(
            self.board, self.option_move_overhead_ms
        )
        send(f"info string chosen stopper {self.active_stopper}")

        self.commands_queue.put(Go(self.board.copy(), self.active_stopper))

    def stop(self, tokens: list[str]) -> None:
        if self.active_stopper and not self.active_stopper.early_stop.is_set():
            self.active_stopper.early_stop.set()

    def quit(self, _: list[str]) -> None:
        self.stop([])
        self.commands_queue.put(Quit())
        self.running = False
