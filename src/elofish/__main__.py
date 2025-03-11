from __future__ import annotations

import argparse
import pathlib
import typing

from elofish.elofish import MatchConfig, run_elo_evaluation

if typing.TYPE_CHECKING:
    Args = argparse.Namespace


def run_cli(args: Args) -> None:
    config = MatchConfig.from_file(args.config)
    run_elo_evaluation(config, enable_game_tick=args.game_tick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("elofish")

    parser.add_argument(
        "config",
        help="Path to config, look to configs/elo_match",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--game-tick",
        help="Prints move and nodes after each move (for debug)",
        action="store_true",
    )

    args = parser.parse_args()
    run_cli(args)
