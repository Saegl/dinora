from __future__ import annotations
import typing
import sys
import traceback
import pathlib
import argparse

from dinora import DEFAULT_WEIGHTS
from dinora.engine import Engine
from dinora.uci.uci import UciState

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinora",
        description="Chess engine",
    )
    parser.add_argument(
        "--model",
        default="alphanet",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Path to model weights",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
    )
    return parser


def run_cli(args: Args) -> None:
    try:
        engine = Engine(args.model, args.weights, args.device)
        uci_state = UciState(engine)
        uci_state.loop()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    except:  # noqa: E722
        with open("dinora.log", "w", encoding="utf8") as logfile:
            exc_type, exc_value, exc_tb = sys.exc_info()
            logfile.write(
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            )
            logfile.write("\n")

        with open("dinora.log", encoding="utf8") as f:
            print(f.read())
