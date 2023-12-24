import pathlib

from dinora import DEFAULT_WEIGHTS


def build_parser(subparsers):
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


def run_cli(args):
    from dinora.onnx.export_onnx import export_onnx

    export_onnx(args.model, args.weights)
