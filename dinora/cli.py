import argparse

import dinora.elo_estimator
from dinora.uci import start_uci


def cli():
    parser = argparse.ArgumentParser(
        prog="dinora",
        description="Chess engine",
    )

    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    elo_estimator = subparsers.add_parser(
        name="elo_estimator", help="Estimate elo of chess engines"
    )
    dinora.elo_estimator.init_cli(elo_estimator)

    args = parser.parse_args()

    if args.subcommand == "elo_estimator":
        dinora.elo_estimator.run_cli(args)
    else:
        start_uci()
