import pathlib

from dinora import DEFAULT_WEIGHTS
from dinora.cli_utils import Subparsers, Args


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
