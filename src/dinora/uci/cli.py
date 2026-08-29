from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
import typing
from dataclasses import dataclass, field

from dinora.engine import Engine
from dinora.models.registry import MODELS
from dinora.options import render_options, uci_options
from dinora.search.registry import DEFAULT_SEARCHER, SEARCHERS, build_searcher
from dinora.threads import limit_threads
from dinora.uci.uci import EngineParams, uci_start

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
        # `--help` is handled in `main`, so that the searcher is known by then
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help, with the UCI options of the selected searcher",
    )
    parser.add_argument(
        "--searcher",
        choices=sorted(SEARCHERS),
        help="Name of the searcher to use",
    )
    parser.add_argument(
        "--model",
        help=f"Name of the model to use ({', '.join(sorted(MODELS))})",
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


def print_help(parser: argparse.ArgumentParser, searcher_name: str) -> None:
    """
    Print `--help` plus the UCI options of `searcher_name`.

    Building the searcher imports numpy & co, so this must stay out of the
    parsing path, see `dinora.threads`.
    """
    others = [name for name in sorted(SEARCHERS) if name != searcher_name]

    parser.epilog = "\n".join(
        [
            "Options are sent by the GUI as `setoption name <name> value <value>`.",
            "",
            "Engine options:",
            render_options(uci_options(EngineParams())),
            "",
            f"UCI options of searcher `{searcher_name}`:",
            render_options(uci_options(build_searcher(searcher_name).params)),
            "",
            f"Options differ per searcher. Others: {', '.join(others)}",
            f"Run `{parser.prog} --searcher {others[0]} --help` to see theirs.",
        ]
    )
    parser.print_help()


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

    if args.help:
        searcher_name = args.searcher or default_args.searcher or DEFAULT_SEARCHER
        print_help(parser, searcher_name)
        return

    run_cli(args, default_args)
