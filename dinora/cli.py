import pathlib
import argparse

from dinora.uci import start_uci
import dinora.elo_estimator.cli as elo_estimator_cli
import dinora.viz.cli as treeviz_cli


def run_cli():
    parser = build_root_cli()
    args = parser.parse_args()

    if args.subcommand == "elo_estimator":
        elo_estimator_cli.run_cli(args)
    elif args.subcommand == "treeviz":
        treeviz_cli.run_cli(args)
    else:
        start_uci(args.model, args.weights, args.device)


def build_root_cli():
    parser = build_uci_cli_parser()
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")
    elo_estimator_cli.build_parser(subparsers)
    treeviz_cli.build_parser(subparsers)
    return parser


def build_uci_cli_parser():
    parser = argparse.ArgumentParser(
        prog="dinora",
        description="Chess engine",
    )
    parser.add_argument(
        "--model",
        default="alphanet",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--weights",
        default="models/model021.ckpt",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    return parser
