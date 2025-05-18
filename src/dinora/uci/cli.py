from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
import typing
from dataclasses import dataclass, field

from dinora.engine import Engine
from dinora.uci.uci import uci_start

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


@dataclass
class DefaultArgs:
    searcher: str | None = field(default=None)
    model: str | None = field(default=None)
    device: str | None = field(default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinora",
        description="Chess engine",
    )
    parser.add_argument(
        "--searcher",
        help="Name of the searcher to use",
    )
    parser.add_argument(
        "--model",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--weights",
        help="Path to model weights",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
        help="Model device, `cpu` or `cuda`",
    )
    return parser


def run_cli(args: Args, default_args: DefaultArgs) -> None:
    try:
        engine = Engine(
            args.searcher or default_args.searcher,
            args.model or default_args.model,
            args.weights,
            args.device or default_args.device,
        )
        uci_start(engine)

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


def main(default_args: DefaultArgs) -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args, default_args)
