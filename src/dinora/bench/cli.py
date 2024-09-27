from __future__ import annotations

import argparse
import pathlib
import typing

from dinora.bench.selfplay import selfplay

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def build_parser(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(name="bench", help="Benchmark speed of engine")
    parser.add_argument(
        "--model",
        default="alphanet",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--weights",
        help="Path to model weights",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
        type=str,
    )


def run_cli(args: Args) -> None:
    selfplay(args.model, args.weights, args.device)
