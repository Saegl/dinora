import pathlib
import chess

from dinora import DEFAULT_WEIGHTS
from dinora.viz.treeviz import render_state, RenderParams, DEFAULT_OUTPUT_DIR
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
    parser.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
        help="Chess FEN of root node",
    )
    parser.add_argument(
        "--nodes",
        default=15,
        help="Number of nodes to search by engine",
        type=int,
    )
    parser.add_argument(
        "--render-nodes",
        default=10,
        help="Number of nodes to render",
        type=int,
    )
    parser.add_argument(
        "--disable-other-node",
        help="`Other node` shows statistics of non rendered nodes",
        action="store_true",
    )
    parser.add_argument(
        "--disable-prior",
        help="Disable priors on arrows between nodes",
        action="store_true",
    )
    parser.add_argument(
        "--disable-gui",
        help="When not disabled, rendered tree will be opened"
        " in default OS apps: default browser (Google Chrome) or photos app",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for generated images",
        default=DEFAULT_OUTPUT_DIR,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--imgformat",
        default="svg",
        help="Output image format svg/png",
    )


def run_cli(args):
    engine = Engine(args.model, args.weights, args.device)

    render_state(
        engine,
        fen=args.fen,
        nodes=args.nodes,
        render_params=RenderParams(
            render_nodes=args.render_nodes,
            show_other_node=not args.disable_other_node,
            show_prior=not args.disable_prior,
            open_default_gui=not args.disable_gui,
            output_dir=args.output_dir,
            imgformat=args.imgformat,
        ),
    )
