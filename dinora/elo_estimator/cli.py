import json
import pathlib

from dinora.elo_estimator import glicko2
from dinora.elo_estimator.elo_estimator import load_players, play_match


def build_parser(subparsers):
    parser = subparsers.add_parser(
        name="elo_estimator", help="Estimate elo of chess engines"
    )
    parser.add_argument(
        "config",
        help="Path to config, look to configs/elo_match",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--output",
        help="Output pgn filename, will be autogenerated if not provided",
    )


def run_cli(args):
    with args.config.open(encoding="utf8") as f:
        config = json.load(f)

    env = glicko2.Glicko2()
    teacher_player, student_player = load_players(config)

    if args.output:
        pgn_output = pathlib.Path(args.output)
    else:
        pgn_output = pathlib.Path(f"{teacher_player.name}_vs_{student_player.name}.pgn")

    with pgn_output.open("wt", encoding="utf8") as output:
        for game in play_match(
            env,
            student_player,
            teacher_player,
            max_games=config["max_games"],
            min_phi=config["min_phi"],
            min_mu=config["min_mu"],
        ):
            print(game, end="\n\n", flush=True)
            print(game, end="\n\n", flush=True, file=output)
