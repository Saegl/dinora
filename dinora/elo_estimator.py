import datetime
import itertools
import random

import chess
import chess.pgn
import chess.engine

from dinora import glicko2
from dinora.glicko2 import Glicko2
from dinora.engine import Engine
from dinora.mcts.constraints import Constraint, NodesCountConstraint


DEFAULT_MAX_GAMES = 100
DEFAULT_MIN_PHI = 75.0


def clip(minval: int, x: int, maxval: int) -> int:
    return min(maxval, max(minval, x))


class RatedPlayer:
    name: str

    def init_rating(self, env: Glicko2) -> None:
        self.rating = env.create_rating()

    def play(self, board: chess.Board) -> chess.Move:
        pass

    def close(self) -> None:
        pass


class TeacherPlayer(RatedPlayer):
    def set_similar_strength(self, other: RatedPlayer, env: Glicko2) -> None:
        pass


class StockfishPlayer(TeacherPlayer):
    STOCKFISH_MIN_ELO: int = 1320
    STOCKFISH_MAX_ELO: int = 3190

    def __init__(self, env: Glicko2, command: str, limit: chess.engine.Limit):
        self.init_rating(env)
        self.limit = limit
        self.uci_engine = chess.engine.SimpleEngine.popen_uci(command)
        self.uci_engine.configure({"UCI_LimitStrength": True})
        self.name = command

    def play(self, board: chess.Board) -> chess.Move:
        playres = self.uci_engine.play(board, limit=self.limit)
        assert playres.move
        return playres.move

    def set_elo(self, target_elo: int) -> None:
        self.uci_engine.configure({"UCI_Elo": target_elo})

    def set_similar_strength(self, other: RatedPlayer, env: Glicko2) -> None:
        target_elo = clip(
            StockfishPlayer.STOCKFISH_MIN_ELO,
            int(other.rating.mu),
            StockfishPlayer.STOCKFISH_MAX_ELO,
        )
        self.rating = env.create_rating(target_elo)
        self.set_elo(target_elo)

    def close(self) -> None:
        self.uci_engine.close()


class DinoraPlayer(RatedPlayer):
    def __init__(self, env: Glicko2, limit: Constraint) -> None:
        self.init_rating(env)
        self.engine = Engine("alphanet")
        self.engine.load_model()
        self.name = "dinora"
        self.limit = limit
        ############# CPU FIX
        assert self.engine._model
        self.engine._model = self.engine._model.to("cpu")
        self.engine.mcts_params.send_func = lambda _: None

    def play(self, board: chess.Board) -> chess.Move:
        return self.engine.get_best_move(board, self.limit)


def play_game(
    white_player: RatedPlayer,
    black_player: RatedPlayer,
    players: tuple[RatedPlayer, RatedPlayer],
    dinora_player: RatedPlayer,
    game_ind: int,
) -> chess.pgn.Game:
    board = chess.Board()
    game = chess.pgn.Game(
        headers={
            "Event": "Elo estimate",
            "Site": "Dinora elo_estimator.py",
            "Stage": f"Dinora phi: {int(dinora_player.rating.phi)}",
            "Date": datetime.date.today().strftime(r"%Y.%m.%d"),
            "White": white_player.name,
            "Black": black_player.name,
            "Round": str(game_ind),
            "WhiteElo": str(int(white_player.rating.mu)),
            "BlackElo": str(int(black_player.rating.mu)),
        }
    )
    node: chess.pgn.GameNode = game

    for player in itertools.cycle(players):
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
    env: Glicko2,
    student_player: RatedPlayer,
    teacher_player: TeacherPlayer,
    max_games: int = DEFAULT_MAX_GAMES,
    min_phi: float = DEFAULT_MIN_PHI,
) -> None:
    game_ind = 0

    while student_player.rating.phi > min_phi and game_ind < max_games:
        teacher_player.set_similar_strength(student_player, env)

        players = random.choice(
            [(student_player, teacher_player), (teacher_player, student_player)]
        )
        white_player = players[0]
        black_player = players[1]

        game = play_game(white_player, black_player, players, student_player, game_ind)
        result = game.headers["Result"]

        if (
            result == "1-0"
            and white_player == student_player
            or result == "0-1"
            and black_player == student_player
        ):
            dinora_outcome = glicko2.WIN
        elif result == "1/2-1/2":
            dinora_outcome = glicko2.DRAW
        else:
            dinora_outcome = glicko2.LOSS

        student_player.rating = env.rate(
            student_player.rating, [(dinora_outcome, teacher_player.rating)]
        )

        print(game, end="\n\n", flush=True)

        game_ind += 1

    for player in players:
        player.close()


if __name__ == "__main__":
    env = Glicko2()
    # student_player = DinoraPlayer(env, limit=NodesCountConstraint(3))

    student_player = StockfishPlayer(
        env, "stockfish", limit=chess.engine.Limit(nodes=3)
    )
    student_player.set_elo(2000)

    teacher_player = StockfishPlayer(
        env, "stockfish", limit=chess.engine.Limit(nodes=3)
    )

    play_match(env, student_player, teacher_player)
