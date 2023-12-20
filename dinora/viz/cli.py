import pathlib
import chess

from dinora import DEFAULT_WEIGHTS
from dinora.viz.treeviz import render_state, RenderParams
from dinora.engine import Engine


def build_parser(subparsers):
    parser = subparsers.add_parser(name="treeviz", help="Render MCTS tree")
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
        default="cuda",
    )


def run_cli(args):
    engine = Engine(args.model, args.weights, args.device)

    render_state(
        engine,
        chess.STARTING_FEN,
        nodes=100,
        format="svg",
        render_params=RenderParams(
            max_number_of_nodes=150,
            open_default_gui=True,
        ),
    )
