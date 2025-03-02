import multiprocessing as mp
import os
import pathlib
import time
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt

from dinora.encoders.board_tensor import boards_to_tensor
from dinora.encoders.policy import legal_policy
from dinora.models.alphanet import AlphaNet
from dinora.models.base import Priors, StateValue
from dinora.search.mcts import mcts
from dinora.search.noise import apply_noise
from rl.replay_buffer import ReplayBuffer

npf32 = npt.NDArray[np.float32]

GPU_WORKER_LOG_TEMPLATE = """\
GPU WORKER INFO {cuda_device}
{timers}
GPU Utilization: {gpu_util:.3f}%
Number of batch calls: {batch_calls}
Number of generated games {completed_games} / {games_count}
Batch speed: {batch_speed:.3f} batches/second
Positions speed: {pos_speed:.3f} pos/second
Game generation speed: {game_speed:.3f} games/second
================================================================================\
"""

CPU_WORKER_LOG_TEMPLATE = """\
CPU WORKER INFO {batch_worker_id}
Plies {plies}
{timers}
================================================================================\
"""


class Timers:
    def __init__(self, log_interval: int, *names: str):
        self.total_time = {}
        self.log_interval = log_interval
        for name in names:
            self.add(name)
        self.last_log_time = time.time()

    def add(self, name):
        self.total_time[name] = 0.0

    @contextmanager
    def timing_section(self, name: str):
        start_time = time.time()
        yield
        time_took = time.time() - start_time
        self.total_time[name] += time_took

    def log_interval_tick(self):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = time.time()
            return True
        else:
            return False

    def dump(self) -> str:
        lines = []
        for name, value in self.total_time.items():
            lines.append(f"{name} time: {value:.3f} seconds")
        return "\n".join(lines)

    def reset(self):
        self.total_time = {name: 0.0 for name in self.total_time}


def sample_softmax_move(node: mcts.Node) -> chess.Move:
    moves = list(node.children.keys())
    visits = np.array(
        [child.visits for child in node.children.values()], dtype=np.float64
    )

    if len(moves) == 0:
        raise ValueError("No available moves to sample from.")

    # If all visits are zero, fallback to uniform probabilities
    if visits.sum() == 0:
        probs = np.ones_like(visits) / len(visits)
    else:
        temperature = 1.0
        adjusted_visits = visits ** (1.0 / temperature)
        probs = adjusted_visits / adjusted_visits.sum()

    move = np.random.choice(moves, p=probs)  # type: ignore
    return move


class Game:
    def __init__(self, nodes_per_move: int, cpuct: float, opening_noise_moves: int):
        self.root = None
        self.leaf = None
        self.board = chess.Board()
        self.played_moves = []
        self.nodes_per_move = nodes_per_move
        self.cpuct = cpuct
        self.opening_noise_moves = opening_noise_moves

    def advance(self):
        assert self.root
        if self.board.ply() < 2 * self.opening_noise_moves:
            move = sample_softmax_move(self.root)
        else:
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
        priors: Priors,
        value: StateValue,
        dirichlet_alpha: float,
        noise_eps: float,
    ):
        if self.root is None:
            self.root = mcts.Node(None, value, 1.0, chess.Move.null())
            priors = apply_noise(
                priors,
                dirichlet_alpha=dirichlet_alpha,
                noise_eps=noise_eps,
            )
            mcts.expand(self.root, priors)
            return

        assert self.leaf
        mcts.expand(self.leaf, priors)
        mcts.backup(self.leaf, self.board, value)


class GamesBatch:
    def __init__(
        self,
        batch_size: int,
        nodes_per_move: int,
        completed_games,
        search_cfg: mcts.MctsParams,
        games_queue: mp.Queue,
    ):
        self.nodes_per_move = nodes_per_move
        self.games = [
            Game(nodes_per_move, search_cfg.cpuct, search_cfg.opening_noise_moves)
            for _ in range(batch_size)
        ]
        self.completed_games = completed_games
        self.search_cfg = search_cfg
        self.games_queue = games_queue

    def gather_batch(self):
        for i in range(len(self.games)):
            game = self.games[i]
            done = game.next()
            if done:
                self.completed_games.value += 1
                self.games_queue.put(game.played_moves)
                self.games[i] = Game(
                    self.nodes_per_move,
                    self.search_cfg.cpuct,
                    self.search_cfg.opening_noise_moves,
                )
                game = self.games[i]
                game.next()
        batch = [game.board for game in self.games]
        return batch

    def backprop(self, evals):
        for i in range(len(self.games)):
            game = self.games[i]
            priors, value = evals[i]
            game.submit(
                priors,
                value,
                self.search_cfg.dirichlet_alpha,
                self.search_cfg.noise_eps,
            )


def database_worker(
    games_count: int,
    completed_games,  # mp.Value
    replay_buffer: ReplayBuffer,
    games_queue: mp.Queue,
):
    while completed_games.value < games_count:
        moves = games_queue.get()
        replay_buffer.add_game(moves)


def cpu_worker(
    batch_worker_id: int,
    batch_size: int,
    nodes_per_move: int,
    batch_queue: mp.Queue,
    eval_queue: mp.Queue,
    completed_games,  # mp.Value
    search_cfg: mcts.MctsParams,
    log_interval: int,
    shared_board_tensor: npf32,
    raw_policy: npf32,
    raw_value: npf32,
    games_queue: mp.Queue,
):
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    games_batch = GamesBatch(
        batch_size, nodes_per_move, completed_games, search_cfg, games_queue
    )
    timers = Timers(log_interval, "gather", "put", "get", "backprop", "decode")

    while True:
        with timers.timing_section("gather"):
            boards = games_batch.gather_batch()
            board_tensor = boards_to_tensor(boards)

        with timers.timing_section("put"):
            shared_board_tensor[:] = board_tensor[:]
            batch_queue.put(batch_worker_id)

        with timers.timing_section("get"):
            eval_queue.get()

        with timers.timing_section("decode"):
            evals = [
                (legal_policy(raw_policy[i], board), float(raw_value[i, 0]))
                for i, board in enumerate(boards)
            ]

        with timers.timing_section("backprop"):
            games_batch.backprop(evals)

        if batch_worker_id == 0 and timers.log_interval_tick():
            print(
                CPU_WORKER_LOG_TEMPLATE.format(
                    batch_worker_id=batch_worker_id,
                    plies=[game.board.ply() for game in games_batch.games],
                    timers=timers.dump(),
                ),
                flush=True,
            )
            timers.reset()


def gpu_worker(
    device: str,
    batch_queue: mp.Queue,
    eval_queues: list[mp.Queue],
    model: AlphaNet,
    completed_games,  # mp.Value
    games_count: int,
    log_interval: int,
    shared_boards_tensors: npf32,
    shared_policy_tensors: npf32,
    shared_value_tensors: npf32,
):
    model = model.to(device)
    timers = Timers(log_interval, "get", "inference", "put")

    batch_calls = 0
    positions = 0

    while True:
        with timers.timing_section("get"):
            batch_worker_id = batch_queue.get()
            input_np = shared_boards_tensors[batch_worker_id]

        with timers.timing_section("inference"):
            raw_policy, raw_value = model.inference_np(input_np)

            batch_calls += 1
            positions += input_np.shape[0]

        with timers.timing_section("put"):
            shared_policy_tensors[batch_worker_id][:] = raw_policy
            shared_value_tensors[batch_worker_id][:] = raw_value
            eval_queues[batch_worker_id].put(1)

        if timers.log_interval_tick():
            times_sum = sum(timers.total_time.values()) + 0.0000001
            print(
                GPU_WORKER_LOG_TEMPLATE.format(
                    cuda_device=device,
                    timers=timers.dump(),
                    gpu_util=(timers.total_time["inference"] / times_sum) * 100,
                    batch_calls=batch_calls,
                    completed_games=completed_games.value,
                    games_count=games_count,
                    batch_speed=batch_calls / times_sum,
                    pos_speed=positions / times_sum,
                    # TODO: divide by total_times_sum
                    game_speed=completed_games.value / times_sum,
                ),
                flush=True,
            )

            batch_calls = 0
            positions = 0
            timers.reset()


def selfplay(
    model: AlphaNet,
    games_count: int,
    nodes_per_move: int,
    batch_size: int,
    cpuct: float,
    opening_noise_moves: int,
    dirichlet_alpha: float,
    noise_eps: float,
    replay_buffer: ReplayBuffer,
    log_interval: int,
    num_batch_workers: int,
    cuda_devices: list[str],
):
    mp.set_start_method("spawn", force=True)  # Needed for proper cuda init
    completed_games = mp.Value("i", 0)

    batch_queue = mp.Queue()

    boards_shape = (batch_size, 18, 8, 8)
    policy_shape = (batch_size, 1880)
    value_shape = (batch_size, 1)

    boards_shms = [
        SharedMemory(create=True, size=np.zeros(boards_shape, dtype=np.float32).nbytes)
        for _ in range(num_batch_workers)
    ]

    shared_boards_tensors = [
        np.ndarray(boards_shape, np.float32, buffer=shm.buf) for shm in boards_shms
    ]

    policy_shms = [
        SharedMemory(create=True, size=np.zeros(policy_shape, dtype=np.float32).nbytes)
        for _ in range(num_batch_workers)
    ]

    shared_policy_tensors = [
        np.ndarray(policy_shape, np.float32, buffer=shm.buf) for shm in policy_shms
    ]

    value_shms = [
        SharedMemory(create=True, size=np.zeros(value_shape, dtype=np.float32).nbytes)
        for _ in range(num_batch_workers)
    ]

    shared_value_tensors = [
        np.ndarray(value_shape, np.float32, buffer=shm.buf) for shm in value_shms
    ]

    eval_queues = [mp.Queue() for _ in range(num_batch_workers)]
    search_cfg = mcts.MctsParams(
        cpuct=cpuct,
        opening_noise_moves=opening_noise_moves,
        dirichlet_alpha=dirichlet_alpha,
        noise_eps=noise_eps,
    )
    games_queue = mp.Queue()

    batch_workers = [
        mp.Process(
            target=cpu_worker,
            args=(
                batch_worker_id,
                batch_size,
                nodes_per_move,
                batch_queue,
                eval_queues[batch_worker_id],
                completed_games,
                search_cfg,
                log_interval,
                board_tensor,
                raw_policy,
                raw_value,
                games_queue,
            ),
        )
        for batch_worker_id, board_tensor, raw_policy, raw_value in zip(
            range(num_batch_workers),
            shared_boards_tensors,
            shared_policy_tensors,
            shared_value_tensors,
        )
    ]

    gpu_workers = [
        mp.Process(
            target=gpu_worker,
            args=(
                device,
                batch_queue,
                eval_queues,
                model,
                completed_games,
                games_count,
                log_interval,
                shared_boards_tensors,
                shared_policy_tensors,
                shared_value_tensors,
            ),
        )
        for device in cuda_devices
    ]

    for p in batch_workers + gpu_workers:
        p.start()

    database_worker(games_count, completed_games, replay_buffer, games_queue)

    for p in batch_workers + gpu_workers:
        p.kill()

    for shm in boards_shms + policy_shms + value_shms:
        shm.unlink()

    print("Selfplay generation complete")


def analyze_pgn(pgn_file: pathlib.Path):
    uniq_positions = set()
    uniq_games = set()

    positions_count = 0
    games_count = 0

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

    pgn_output.close()

    print(f"Uniq games {(len(uniq_games) / games_count) * 100:.3f}%")
    print(f"Uniq positions rate {(len(uniq_positions) / positions_count) * 100:.3f}%")
    print(f"Average plies {positions_count / games_count}")
    print("White wins:", white)
    print("Black wins:", black)
    print("Draw:", draw)
    print("Total games", white + black + draw)


if __name__ == "__main__":
    from dinora.models import model_selector

    model_path = pathlib.Path("models/alphanet_mini.ckpt")
    # model_path = pathlib.Path("models/alphanet_classic.ckpt")
    model = model_selector("alphanet", model_path, "cpu")
    pgn_file = pathlib.Path("example.pgn")
    assert isinstance(model, AlphaNet)
    replay_buffer_dir = pathlib.Path("selfplay_rb")
    replay_buffer_dir.mkdir(exist_ok=True)
    selfplay(
        model,
        games_count=128,
        nodes_per_move=100,
        batch_size=16,
        cpuct=3.0,
        opening_noise_moves=15,
        dirichlet_alpha=0.3,
        noise_eps=0.25,
        replay_buffer=ReplayBuffer(replay_buffer_dir, 1000, 1000, 128),
        log_interval=15,
        num_batch_workers=3,
        cuda_devices=["cuda:0"],
    )
    print()
    analyze_pgn(pgn_file)
