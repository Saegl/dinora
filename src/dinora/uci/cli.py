from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
import typing
from dataclasses import dataclass, field

from dinora.engine import Engine
from dinora.search.registry import SEARCHERS
from dinora.threads import limit_threads
from dinora.uci.uci import uci_start

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


@dataclass
class DefaultArgs:
    searcher: str | None = field(default=None)
    model: str | None = field(default=None)
    device: str | None = field(default=None)
    limit_threads: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinora",
        description="Chess engine",
    )
    parser.add_argument(
        "--searcher",
        choices=sorted(SEARCHERS),
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
    parser.add_argument(
        "--limit-threads",
        action="store_true",
        help="Limit ONNX/numpy to 1 thread",
    )
    return parser


def run_cli(args: Args, default_args: DefaultArgs) -> None:
    try:
        should_limit_threads = args.limit_threads or default_args.limit_threads
        if should_limit_threads:
            # Before `Engine`, which is what pulls in numpy/onnxruntime/torch
            limit_threads()

        engine = Engine(
            args.searcher or default_args.searcher,
            args.model or default_args.model,
            args.weights,
            args.device or default_args.device,
            limit_threads=should_limit_threads,
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
