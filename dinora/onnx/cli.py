from __future__ import annotations
import pathlib
import typing
import argparse

from dinora import DEFAULT_WEIGHTS

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def build_parser(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        name="export_onnx", help="Export torch model as onnx"
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


def run_cli(args: Args) -> None:
    from dinora.onnx.export_onnx import export_onnx

    export_onnx(args.model, args.weights)
