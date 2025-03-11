import abc
import dataclasses
import datetime
import itertools
import json
import pathlib
import random
import typing

import chess
import chess.engine
import chess.pgn
import tqdm
from colorama import Fore, just_fix_windows_console

from elofish.glicko2 import glicko2

DEFAULT_MAX_GAMES = 100
DEFAULT_MIN_PHI = 75.0
DEFAULT_MIN_MU = 1200


def clip(minval: int, x: int, maxval: int) -> int:
    return min(maxval, max(minval, x))


class RatedPlayer(abc.ABC):
    rating: glicko2.Rating

    @property
    @abc.abstractmethod
    def fullname(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def play(self, board: chess.Board) -> tuple[chess.Move, int]:
        pass

    @abc.abstractmethod
    def dump_info(self) -> dict[str, str]:
        pass

    @abc.abstractmethod
    def dump_options(self) -> dict[typing.Any, typing.Any]:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class TeacherPlayer(RatedPlayer, abc.ABC):
    @abc.abstractmethod
    def set_similar_strength(self, other: RatedPlayer) -> None:
        pass


class UCIPlayer(RatedPlayer):
    def __init__(
        self,
        rating: glicko2.Rating,
        command: str,
        options: dict[str, str],
        nodes_limit: int | None = None,
        time_limit: float | None = None,  # seconds per move
    ):
        self.rating = rating
        self.command = command
        self.time_limit = time_limit
        self.nodes_limit = nodes_limit
        self.options = options

        self.uci_engine = chess.engine.SimpleEngine.popen_uci(command)
        self.uci_engine.configure(options)

    @property
    def fullname(self) -> str:
        output = self.name
        if self.nodes_limit:
            output += f"_{self.nodes_limit}nodes"
        if self.time_limit:
            output += f"_{self.time_limit}sec_move"
        return output

    @property
    def name(self) -> str:
        return self.uci_engine.id.get("name", "UnknownEngine")

    def play(self, board: chess.Board) -> tuple[chess.Move, int]:
        playres = self.uci_engine.play(
            board,
            info=chess.engine.INFO_BASIC,
            limit=chess.engine.Limit(
                time=self.time_limit,
                nodes=self.nodes_limit,
            ),
        )
        assert playres.move
        return playres.move, playres.info.get("nodes", -1)

    def dump_info(self) -> dict[str, str]:
        return dict(self.uci_engine.id)

    def dump_options(self) -> dict[typing.Any, typing.Any]:
        options = {}
        for k, v in self.uci_engine.options.items():
            options[k] = self.options.get(k, v.default)
        return options

    def close(self) -> None:
        self.uci_engine.close()

    def reset(self) -> None:
        self.uci_engine.close()
        self.uci_engine = chess.engine.SimpleEngine.popen_uci(self.command)
        self.uci_engine.configure(self.options)


class StockfishPlayer(UCIPlayer, TeacherPlayer):
    STOCKFISH_MIN_ELO: int = 1320
    STOCKFISH_MAX_ELO: int = 3190

    def __init__(
        self,
        rating: glicko2.Rating,
        command: str,
        options: dict[str, str],
        nodes_limit: int | None = None,
        time_limit: float | None = None,  # seconds per move
        elo: int | None = None,
    ):
        super().__init__(rating, command, options, nodes_limit, time_limit)

        if elo:
            self.set_elo(elo)

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


def play_game(
    white_player: RatedPlayer,
    black_player: RatedPlayer,
    student_player: RatedPlayer,
    game_ind: int,
    game_tick: bool,
) -> chess.pgn.Game:
    board = chess.Board()
    current_datetime = datetime.datetime.now()
    utc_datetime = datetime.datetime.now(datetime.timezone.utc)
    game = chess.pgn.Game(
        headers={
            "Event": "Elo estimate",
            "Site": "Dinora elofish.py",
            "Stage": f"{student_player.fullname} phi: {int(student_player.rating.phi)}",
            "Date": current_datetime.date().strftime(r"%Y.%m.%d"),
            "UTCDate": utc_datetime.date().strftime(r"%Y.%m.%d"),
            "Time": current_datetime.strftime("%H:%M:%S"),
            "UTCTime": utc_datetime.strftime("%H:%M:%S"),
            "White": white_player.fullname,
            "Black": black_player.fullname,
            "Round": str(game_ind),
            "WhiteElo": str(int(white_player.rating.mu)),
            "BlackElo": str(int(black_player.rating.mu)),
            "StudentRatingDeviation": f"{student_player.rating.phi:.2f}",
        }
    )
    node: chess.pgn.GameNode = game

    teacher_total_nodes = 0
    teacher_plies = 0

    student_total_nodes = 0
    student_plies = 0

    for player in itertools.cycle([white_player, black_player]):
        if not board.outcome(claim_draw=True):
            move, nodes = player.play(board)
            node = node.add_variation(move)
            board.push(move)
            if game_tick:
                print(move.uci(), nodes)

            if player is student_player:
                student_total_nodes += nodes
                student_plies += 1
            else:
                teacher_total_nodes += nodes
                teacher_plies += 1
        else:
            break

    result = board.result(claim_draw=True)
    outcome = board.outcome(claim_draw=True)
    game.headers["Result"] = result
    game.headers["TerminationEnum"] = (
        str(outcome.termination.name) if outcome else "UNKNOWN"
    )
    game.headers["AvgTeacherNodes"] = str(teacher_total_nodes // teacher_plies)
    game.headers["AvgStudentNodes"] = str(student_total_nodes // student_plies)

    white_player.reset()
    black_player.reset()

    return game


def play_match(
    env: glicko2.Glicko2,
    student_player: RatedPlayer,
    teacher_player: TeacherPlayer,
    max_games: int = DEFAULT_MAX_GAMES,
    min_phi: float = DEFAULT_MIN_PHI,
    min_mu: float = DEFAULT_MIN_MU,
    game_tick: bool = False,
) -> typing.Iterator[chess.pgn.Game]:
    game_ind = 0

    while (
        student_player.rating.mu > min_mu
        and student_player.rating.phi > min_phi
        and game_ind < max_games
    ):
        teacher_player.set_similar_strength(student_player)

        white_player, black_player = random.choice(
            [(student_player, teacher_player), (teacher_player, student_player)]
        )

        game = play_game(
            white_player, black_player, student_player, game_ind, game_tick
        )
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

        student_player.rating = env.rate(  # type: ignore
            student_player.rating, [(student_outcome, teacher_player.rating)]
        )
        yield game

        game_ind += 1

    for player in [teacher_player, student_player]:
        player.close()


PLAYER_CLASSES = {
    "StockfishPlayer": StockfishPlayer,
    "UCIPlayer": UCIPlayer,
}


@dataclasses.dataclass
class Rating:
    deviation: float
    rating: float | None = None

    @staticmethod
    def from_dict(d: dict[str, typing.Any]) -> "Rating":
        rating_conf = Rating(deviation=d["phi"], rating=d.get("mu"))
        return rating_conf

    def to_dict(self) -> dict[str, typing.Any]:
        d = {"phi": self.deviation}
        if self.rating is not None:
            d["mu"] = self.rating
        return d


@dataclasses.dataclass
class PlayerConfig:
    player_class: str
    start_rating: Rating
    init: dict

    @staticmethod
    def from_dict(d: dict[str, typing.Any]) -> "PlayerConfig":
        player_conf = PlayerConfig(
            player_class=d["class"],
            start_rating=Rating.from_dict(d["start_rating"]),
            init=d["init"],
        )
        return player_conf

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "class": self.player_class,
            "start_rating": self.start_rating.to_dict(),
            "init": self.init,
        }

    def load_player(self):
        PlayerClass = PLAYER_CLASSES[self.player_class]
        if self.start_rating.rating is not None:
            rating = glicko2.Rating(
                phi=self.start_rating.deviation,  # type: ignore
                mu=self.start_rating.rating,  # type: ignore
            )
        else:
            rating = glicko2.Rating(phi=int(self.start_rating.deviation))
        player = PlayerClass(rating, **self.init)
        return player


@dataclasses.dataclass
class MatchConfig:
    max_games: int
    min_phi: float
    min_mu: float
    teacher_player: PlayerConfig
    student_player: PlayerConfig

    @staticmethod
    def from_file(path: pathlib.Path):
        with path.open("r") as f:
            config = MatchConfig.from_dict(json.load(f))
        return config

    @staticmethod
    def from_dict(d: dict[str, typing.Any]) -> "MatchConfig":
        config = MatchConfig(
            max_games=d["max_games"],
            min_phi=d["min_phi"],
            min_mu=d["min_mu"],
            teacher_player=PlayerConfig.from_dict(d["teacher_player"]),
            student_player=PlayerConfig.from_dict(d["student_player"]),
        )
        return config

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "max_games": self.max_games,
            "min_phi": self.min_phi,
            "min_mu": self.min_mu,
            "teacher_player": self.teacher_player.to_dict(),
            "student_player": self.student_player.to_dict(),
        }


@dataclasses.dataclass
class EvaluationResult:
    new_rating: int
    new_deviation: int
    report_dir: pathlib.Path


def run_elo_evaluation(config: MatchConfig, enable_game_tick=False) -> EvaluationResult:
    just_fix_windows_console()

    env = glicko2.Glicko2()  # type: ignore
    teacher_player: StockfishPlayer = config.teacher_player.load_player()
    student_player: UCIPlayer = config.student_player.load_player()

    start_datetime = datetime.datetime.now(datetime.timezone.utc)
    dir_name = f"{start_datetime.strftime('%Y-%m-%d %H:%M')} {student_player.name}"

    output_dir = pathlib.Path("reports") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pgn_file = output_dir / "game.pgn"
    pgn_output = pgn_file.open("wt", encoding="utf8")

    report_file = output_dir / "report.txt"
    report_output = report_file.open("wt", encoding="utf8")

    logs_file = output_dir / "logs.txt"
    logs_output = logs_file.open("wt", encoding="utf8")

    config_file = output_dir / "config.json"
    with config_file.open("w") as f:
        json.dump(config.to_dict(), f, indent=4)

    options_file = output_dir / "options.json"
    json.dump(
        [
            teacher_player.dump_info(),
            teacher_player.dump_options(),
            student_player.dump_info(),
            student_player.dump_options(),
        ],
        options_file.open("wt", encoding="utf8"),
    )

    report_output.write(f"Start date {start_datetime.strftime('%Y-%m-%d')}\n")
    report_output.write(f"Start time {start_datetime.strftime('%H:%M')}\n")
    report_output.write(
        f"Initial Rating {student_player.rating.mu} ({student_player.rating.phi})\n"
    )

    wins = 0
    draws = 0
    losses = 0

    try:
        for game in tqdm.tqdm(
            play_match(
                env,
                student_player,
                teacher_player,
                max_games=config.max_games,
                min_phi=config.min_phi,
                min_mu=config.min_mu,
                game_tick=enable_game_tick,
            )
        ):
            # print(game, end="\n\n", flush=True)
            print(game, end="\n\n", flush=True, file=pgn_output)

            round_ind = game.headers["Round"]
            elo = game.headers[
                "WhiteElo"
            ]  # Doesn't matter white or black since Teacher copycats Student
            student_rating_deviation = game.headers["StudentRatingDeviation"]
            student_nodes = game.headers["AvgStudentNodes"]
            teacher_nodes = game.headers["AvgTeacherNodes"]

            result = game.headers["Result"]
            student_is_white = game.headers["White"] == student_player.fullname

            if (
                student_is_white
                and result == "1-0"
                or not student_is_white
                and result == "0-1"
            ):
                result_string = f"{Fore.GREEN}Win{Fore.RESET}"
                wins += 1
            elif (
                student_is_white
                and result == "0-1"
                or not student_is_white
                and result == "1-0"
            ):
                result_string = f"{Fore.RED}Loss{Fore.RESET}"
                losses += 1
            else:
                result_string = f"{Fore.YELLOW}Draw{Fore.RESET}"
                draws += 1

            termination = game.headers["TerminationEnum"]

            game_log = (
                f"{Fore.BLUE}{round_ind}{Fore.RESET}:"
                f" {result_string} by {termination}"
                f", Elo = {elo} ({student_rating_deviation})"
                f", {Fore.MAGENTA}StudentSpeed{Fore.RESET} = {student_nodes} n/ply"
                f", {Fore.CYAN}TeacherSpeed{Fore.RESET} = {teacher_nodes} n/ply"
            )

            tqdm.tqdm.write(game_log)
            print(game_log, file=logs_output)

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}Elo estimator early stopping{Fore.RESET}")

    end_datetime = datetime.datetime.now(datetime.timezone.utc)

    report_output.write(f"Wins {wins}\n")
    report_output.write(f"Draws {draws}\n")
    report_output.write(f"Losses {losses}\n")
    report_output.write(f"End time {end_datetime.strftime('%H:%M')}\n")
    report_output.write(f"Time taken {(end_datetime - start_datetime)}\n")
    report_output.write(
        f"Final rating {student_player.rating.mu} ({student_player.rating.phi})\n"
    )

    print(f"{Fore.GREEN}Result saved at {output_dir}{Fore.RESET}")
    return EvaluationResult(
        student_player.rating.mu,
        student_player.rating.phi,
        output_dir,
    )
