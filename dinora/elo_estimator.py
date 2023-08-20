import abc
import datetime
import itertools
import random
import typing
import pathlib
import json

import chess
import chess.pgn
import chess.engine

from dinora import glicko2
from dinora.engine import Engine
from dinora.mcts.constraints import NodesCountConstraint


DEFAULT_MAX_GAMES = 100
DEFAULT_MIN_PHI = 75.0


def clip(minval: int, x: int, maxval: int) -> int:
    return min(maxval, max(minval, x))


class RatedPlayer(abc.ABC):
    name: str
    rating: glicko2.Rating

    @abc.abstractmethod
    def play(self, board: chess.Board) -> chess.Move:
        pass

    def close(self) -> None:
        pass


class TeacherPlayer(RatedPlayer, abc.ABC):
    @abc.abstractmethod
    def set_similar_strength(self, other: RatedPlayer) -> None:
        pass


class StockfishPlayer(TeacherPlayer):
    STOCKFISH_MIN_ELO: int = 1320
    STOCKFISH_MAX_ELO: int = 3190

    def __init__(
        self,
        rating: glicko2.Rating,
        command: str,
        nodes_limit: int,
        elo: int | None = None,
    ):
        self.rating = rating
        self.nodes_limit = nodes_limit
        self.uci_engine = chess.engine.SimpleEngine.popen_uci(command)
        self.name = "stockfish"

        if elo:
            self.set_elo(elo)

    def play(self, board: chess.Board) -> chess.Move:
        playres = self.uci_engine.play(
            board, limit=chess.engine.Limit(nodes=self.nodes_limit)
        )
        assert playres.move
        return playres.move

    @staticmethod
    def clip_elo(target_elo: int) -> int:
        return clip(
            StockfishPlayer.STOCKFISH_MIN_ELO,
            target_elo,
            StockfishPlayer.STOCKFISH_MAX_ELO,
        )

    def set_elo(self, target_elo: int) -> None:
        self.uci_engine.configure(
            {"UCI_LimitStrength": True, "UCI_Elo": StockfishPlayer.clip_elo(target_elo)}
        )

    def set_similar_strength(self, other: RatedPlayer) -> None:
        target_elo = StockfishPlayer.clip_elo(int(other.rating.mu))
        self.rating.mu = target_elo
        self.set_elo(target_elo)

    def close(self) -> None:
        self.uci_engine.close()


class DinoraPlayer(RatedPlayer):
    def __init__(
        self, rating: glicko2.Rating, nodes_limit: int, weights: str, device: str
    ) -> None:
        self.rating = rating
        self.engine = Engine("alphanet", pathlib.Path(weights), device)
        self.engine.load_model()
        self.name = "dinora"
        self.nodes_limit = nodes_limit
        assert self.engine._model
        self.engine.mcts_params.send_func = lambda _: None

    def play(self, board: chess.Board) -> chess.Move:
        return self.engine.get_best_move(board, NodesCountConstraint(self.nodes_limit))


def play_game(
    white_player: RatedPlayer,
    black_player: RatedPlayer,
    student_player: RatedPlayer,
    game_ind: int,
) -> chess.pgn.Game:
    board = chess.Board()
    game = chess.pgn.Game(
        headers={
            "Event": "Elo estimate",
            "Site": "Dinora elo_estimator.py",
            "Stage": f"{student_player.name} phi: {int(student_player.rating.phi)}",
            "Date": datetime.date.today().strftime(r"%Y.%m.%d"),
            "White": white_player.name,
            "Black": black_player.name,
            "Round": str(game_ind),
            "WhiteElo": str(int(white_player.rating.mu)),
            "BlackElo": str(int(black_player.rating.mu)),
        }
    )
    node: chess.pgn.GameNode = game

    for player in itertools.cycle([white_player, black_player]):
        if not board.outcome(claim_draw=True):
            move = player.play(board)
            node = node.add_variation(move)
            board.push(move)
        else:
            break

    result = board.result(claim_draw=True)
    game.headers["Result"] = result

    return game


def play_match(
    env: glicko2.Glicko2,
    student_player: RatedPlayer,
    teacher_player: TeacherPlayer,
    max_games: int = DEFAULT_MAX_GAMES,
    min_phi: float = DEFAULT_MIN_PHI,
) -> typing.Iterator[chess.pgn.Game]:
    game_ind = 0

    while student_player.rating.phi > min_phi and game_ind < max_games:
        teacher_player.set_similar_strength(student_player)

        white_player, black_player = random.choice(
            [(student_player, teacher_player), (teacher_player, student_player)]
        )

        game = play_game(white_player, black_player, student_player, game_ind)
        result = game.headers["Result"]

        if (
            result == "1-0"
            and white_player == student_player
            or result == "0-1"
            and black_player == student_player
        ):
            student_outcome = glicko2.WIN
        elif result == "1/2-1/2":
            student_outcome = glicko2.DRAW
        else:
            student_outcome = glicko2.LOSS

        student_player.rating = env.rate(
            student_player.rating, [(student_outcome, teacher_player.rating)]
        )
        yield game

        game_ind += 1

    for player in [teacher_player, student_player]:
        player.close()


def load_players(
    env: glicko2.Glicko2, config_path: pathlib.Path
) -> tuple[TeacherPlayer, RatedPlayer]:
    with config_path.open(encoding="utf8") as f:
        config = json.load(f)

    if "mu" in config["teacher_player"]["start_rating"]:
        raise ValueError("Cannot set teacher `mu`, it will be similar to student")

    Teacher = globals()[config["teacher_player"]["class"]]
    teacher_init = config["teacher_player"]["init"]
    teacher_rating = glicko2.Rating(phi=config["teacher_player"]["start_rating"]["phi"])
    teacher_player: TeacherPlayer = Teacher(teacher_rating, **teacher_init)

    Student = globals()[config["student_player"]["class"]]
    student_init = config["student_player"]["init"]
    student_start_rating = config["student_player"]["start_rating"]
    student_rating = glicko2.Rating(
        mu=student_start_rating["mu"], phi=student_start_rating["phi"]
    )
    student_player = Student(student_rating, **student_init)

    return teacher_player, student_player


def init_cli(parser):
    parser.add_argument(
        "config",
        help="Path to config, look to configs/elo_match",
        type=pathlib.Path,
    )


def run_cli(args):
    env = glicko2.Glicko2()
    teacher_player, student_player = load_players(env, args.config)

    for game in play_match(env, student_player, teacher_player):
        print(game, end="\n\n", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "elo_estimator",
        description="Estimate elo of chess engines",
    )

    init_cli(parser)
    run_cli(parser.parse_args())
