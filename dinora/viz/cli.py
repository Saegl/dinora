import pathlib
import chess

from dinora.viz.treeviz import render_state, RenderParams
from dinora.mcts.params import MCTSparams


def build_parser(subparsers):
    parser = subparsers.add_parser(name="treeviz", help="Render MCTS tree")
    parser.add_argument(
        "--weights",
        help="Path to model weights",
        type=pathlib.Path,
    )


def run_cli(args):
    import torch

    model = torch.load(args.weights)
    render_state(
        model,
        chess.STARTING_FEN,
        nodes=100,
        format="svg",
        mcts_params=MCTSparams(),
        render_params=RenderParams(
            max_number_of_nodes=150,
            open_default_gui=True,
        ),
    )
