import datetime
import pathlib
import time
from contextlib import contextmanager
from io import TextIOWrapper

import chess
import chess.pgn

from dinora.models.alphanet import AlphaNet
from dinora.search.mcts import mcts
from dinora.search.noise import apply_noise


class Timers:
    def __init__(self):
        self.acc = {}

    def add(self, name):
        self.acc[name] = 0.0

    @contextmanager
    def timing_section(self, name: str):
        start_time = time.time()
        yield
        self.acc[name] += time.time() - start_time


class Game:
    def __init__(self, nodes_per_move: int, cpuct: float):
        self.root = None
        self.leaf = None
        self.board = chess.Board()
        self.played_moves = []
        self.nodes_per_move = nodes_per_move
        self.cpuct = cpuct

    def advance(self):
        assert self.root
        move = mcts.most_visited_move(self.root)
        self.board.push(move)
        self.played_moves.append(move)
        self.root = self.root.children[move]
        self.root.parent = None

    def root_ended(self):
        terminal_value = mcts.terminal_solver(self.board)
        game_too_long = self.board.ply() >= 256 * 2
        return terminal_value is not None or game_too_long

    def next(self) -> bool:
        if self.root is None:
            return False

        if self.root.visits > self.nodes_per_move:
            self.advance()
            if self.root_ended():
                return True

        leaf_selected = False
        while not leaf_selected:
            self.leaf = mcts.select_leaf(self.root, self.board, self.cpuct)
            terminal_value = mcts.terminal_solver(self.board)
            if terminal_value is not None:
                priors, value = {}, terminal_value
                mcts.expand(self.leaf, priors)
                mcts.backup(self.leaf, self.board, value)

                if self.root.visits > self.nodes_per_move:
                    self.advance()
                    if self.root_ended():
                        return True
                    return self.next()
            else:
                leaf_selected = True

        return False

    def submit(
        self,
        priors,
        value,
        opening_noise_moves: int,
        dirichlet_alpha: float,
        noise_eps: float,
    ):
        if self.board.ply() < 2 * opening_noise_moves:
            priors = apply_noise(
                priors,
                dirichlet_alpha=dirichlet_alpha,
                noise_eps=noise_eps,
            )

        if self.root is None:
            self.root = mcts.Node(None, value, 1.0, chess.Move.null())
            mcts.expand(self.root, priors)
            return

        assert self.leaf
        mcts.expand(self.leaf, priors)
        mcts.backup(self.leaf, self.board, value)


def save_game_pgn(game: Game, pgn_output: TextIOWrapper):
    current_datetime = datetime.datetime.now()
    utc_datetime = datetime.datetime.now(datetime.timezone.utc)

    game_pgn = chess.pgn.Game(
        headers={
            "Event": "RL selfplay",
            "Site": "Dinora engine",
            "Date": current_datetime.date().strftime(r"%Y.%m.%d"),
            "UTCDate": utc_datetime.date().strftime(r"%Y.%m.%d"),
            "Time": current_datetime.strftime("%H:%M:%S"),
            "UTCTime": utc_datetime.strftime("%H:%M:%S"),
            # "Round": f"Game number {game_number}",
        }
    )
    node: chess.pgn.GameNode = game_pgn
    board = chess.Board()
    for move in game.played_moves:
        node = node.add_variation(move)
        board.push(move)

    if board.ply() >= 256 * 2:
        result = "1/2-1/2"
    else:
        result = board.result(claim_draw=True)
    game_pgn.headers["Result"] = result
    print(game_pgn, end="\n\n", flush=True, file=pgn_output)


def selfplay(
    model: AlphaNet,
    games_count: int,
    nodes_per_move: int,
    batch_size: int,
    cpuct: float,
    opening_noise_moves: int,
    dirichlet_alpha: float,
    noise_eps: float,
    pgn_file: pathlib.Path,
    log_interval: int,
):
    games = [Game(nodes_per_move, cpuct) for _ in range(batch_size)]
    completed_games = 0

    batch_calls = 0
    positions = 0

    pgn_output = pgn_file.open("w", encoding="utf8")

    last_log_time = time.time()

    timers = Timers()
    timers.add("gather")
    timers.add("inference")
    timers.add("backprop")

    selfplay_start_time = time.time()

    while completed_games < games_count:
        with timers.timing_section("gather"):
            for i in range(batch_size):
                game = games[i]
                done = game.next()
                if done:
                    completed_games += 1
                    save_game_pgn(game, pgn_output)
                    games[i] = Game(nodes_per_move, cpuct)
                    game = games[i]
                    game.next()
            batch = [game.board for game in games]

        with timers.timing_section("inference"):
            outs = model.evaluate_batch(batch)

        with timers.timing_section("backprop"):
            for i in range(batch_size):
                game = games[i]
                game.submit(*outs[i], opening_noise_moves, dirichlet_alpha, noise_eps)

        batch_calls += 1
        positions += batch_size

        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            last_log_time = time.time()
            times_sum = sum(timers.acc.values()) + 0.000001
            print("Batch plies")
            print([game.board.ply() for game in games])
            print(f"Batch gather time: {timers.acc['gather']:.3f}")
            print(f"Inference time: {timers.acc['inference']:.3f}")
            print(f"MCTS backprop time: {timers.acc['backprop']:.3f}")
            print(
                f"GPU Idle {((timers.acc['gather'] + timers.acc['backprop']) / times_sum) * 100:.3f}%"
            )
            print(f"Number of batch calls: {batch_calls}")
            print(f"Number of generated games {completed_games} / {games_count}")
            print(f"Batch speed: {batch_calls / times_sum:.3f} batches/second")
            print(f"Positions speed: {positions / times_sum:.3f} pos/second")
            print(
                f"Game generation speed: {completed_games / times_sum:.3f} games/second"
            )
            print("=" * 80)

    total_time = time.time() - selfplay_start_time

    times_sum = sum(timers.acc.values()) + 0.000001

    print("Selfplay finished")
    print(f"Time taken: {total_time}")
    print(f"Batch gather time: {timers.acc['gather']:.3f}")
    print(f"Inference time: {timers.acc['inference']:.3f}")
    print(f"MCTS backprop time: {timers.acc['backprop']:.3f}")
    print(
        f"GPU Idle {((timers.acc['gather'] + timers.acc['backprop']) / times_sum) * 100:.3f}%"
    )
    print(f"Number of batch calls: {batch_calls}")
    print(f"Number of generated games {completed_games} / {games_count}")
    print(f"Batch speed: {batch_calls / times_sum:.3f} batches/second")
    print(f"Positions speed: {positions / times_sum:.3f} pos/second")
    print(f"Game generation speed: {completed_games / times_sum:.3f} games/second")
    print("=" * 80)

    pgn_output.close()


def analyze_pgn(pgn_file: pathlib.Path):
    uniq_positions = set()
    uniq_games = set()

    positions_count = 0
    games_count = 0
    # TODO: MIN, MAX, AVG moves

    white = 0
    black = 0
    draw = 0

    pgn_output = pgn_file.open("r", encoding="utf8")

    while game := chess.pgn.read_game(pgn_output):
        if game.headers["Result"] == "1-0":
            white += 1
        elif game.headers["Result"] == "0-1":
            black += 1
        else:
            draw += 1

        games_count += 1

        board = chess.Board()
        first_moves = []

        ply_count = 0

        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            uniq_positions.add(board.fen())
            ply_count += 1

            if i < 5:
                first_moves.append(move.uci())

        positions_count += ply_count

        game_hash = "".join(first_moves) + board.fen()
        uniq_games.add(game_hash)

    print(f"Uniq games {(len(uniq_games) / games_count) * 100:.3f}%")
    print(f"Uniq positions rate {(len(uniq_positions) / positions_count) * 100:.3f}%")
    print("White wins:", white)
    print("Black wins:", black)
    print("Draw:", draw)
    print("Total games", white + black + draw)


if __name__ == "__main__":
    from dinora.models import model_selector

    model_path = pathlib.Path("models/alphanet_mini.ckpt")
    model = model_selector("alphanet", model_path, "cuda")
    pgn_file = pathlib.Path("example.pgn")
    assert isinstance(model, AlphaNet)
    selfplay(
        model,
        games_count=64,
        nodes_per_move=80,
        batch_size=64,
        cpuct=3.0,
        opening_noise_moves=10,
        dirichlet_alpha=0.1,
        noise_eps=0.3,
        pgn_file=pgn_file,
        log_interval=5,
    )
    print()
    analyze_pgn(pgn_file)
