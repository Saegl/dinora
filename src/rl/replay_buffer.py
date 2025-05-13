import math
import random
from io import TextIOWrapper
from pathlib import Path

import chess
import chess.pgn
import wandb

from dataset.make import convert_dir as dataset_convert_dir
from train.datamodules import CompactDataModule


def save_game_to_pgn(played_moves: list[chess.Move], pgn_output: TextIOWrapper) -> None:
    game_pgn = chess.pgn.Game(
        headers={
            "Event": "RL selfplay",
        }
    )
    node: chess.pgn.GameNode = game_pgn
    board = chess.Board()
    for move in played_moves:
        node = node.add_variation(move)
        board.push(move)

    if board.ply() >= 256 * 2:
        result = "1/2-1/2"
    else:
        result = board.result(claim_draw=True)
    game_pgn.headers["Result"] = result
    print(game_pgn, end="\n\n", flush=True, file=pgn_output)


def read_game(f: TextIOWrapper) -> list[str]:
    game_source = []

    break_on_next = False
    for line in f:
        game_source.append(line)
        if break_on_next:
            break
        if line.startswith("1."):
            break_on_next = True

    return game_source


class ReplayBuffer:
    def __init__(
        self,
        workdir: Path,
        pgn_chunk_games_size: int,
        window_games_size: int,
        batch_size: int,
        upload_pgn: bool,
    ):
        self.workdir = workdir
        self.pgn_chunk_games_size = pgn_chunk_games_size
        self.window_games_size = window_games_size
        self.batch_size = batch_size
        self.upload_pgn = upload_pgn

        self.chunk_names: list[str] = []
        self.new_chunk()

    def new_chunk(self) -> None:
        self.current_chunk_name = f"{len(self.chunk_names)}.pgn"
        self.current_chunk_games = 0
        self.current_chunk_path = self.workdir / self.current_chunk_name
        self.current_chunk_file = self.current_chunk_path.open("w")
        self.chunk_names.append(self.current_chunk_name)

    def prepare_dataset(self) -> CompactDataModule:
        self.current_chunk_file.close()

        num_chunks = math.ceil(self.window_games_size / self.pgn_chunk_games_size)

        start_index = max(0, len(self.chunk_names) - num_chunks)
        window_chunks = self.chunk_names[start_index:]

        piles_dir = self.workdir / "piles"
        piles_dir.mkdir(exist_ok=True)
        for file in piles_dir.iterdir():
            if file.is_file():
                file.unlink()

        piles = [piles_dir / f"{i}.pgn" for i in range(len(window_chunks))]
        for pile in piles:
            pile.touch()

        print("Chunks for dataset", window_chunks)

        for window_chunk in window_chunks:
            window_chunk_path = self.workdir / window_chunk
            with window_chunk_path.open("r") as wf:
                while True:
                    game_source = read_game(wf)
                    if len(game_source) == 0:
                        break

                    random_pile = random.choice(piles)
                    with random_pile.open("a") as pf:
                        pf.writelines(game_source)

        dataset_path_dir = self.workdir / "dataset"
        dataset_path_dir.mkdir(exist_ok=True)
        for file in dataset_path_dir.iterdir():
            if file.is_file():
                file.unlink()

        dataset_convert_dir(
            piles_dir,
            dataset_path_dir,
            files_count=None,
            q_nodes=0,
            train_percentage=1.0,
            val_percentage=0.0,
            test_percentage=0.0,
        )

        self.current_chunk_file = self.current_chunk_path.open("a")

        datamodule = CompactDataModule(
            dataset_path_dir, z_weight=1.0, q_weight=0.0, batch_size=self.batch_size
        )
        print("Batches in replaybuffer:", len(datamodule.train_dataloader()))
        return datamodule

    def add_game(self, moves: list[chess.Move]) -> None:
        # print("Writing game to ", self.current_chunk_path)
        save_game_to_pgn(moves, self.current_chunk_file)
        self.current_chunk_games += 1

        if self.current_chunk_games >= self.pgn_chunk_games_size:
            self.current_chunk_file.close()

            if self.upload_pgn:
                artifact = wandb.Artifact("selfplay_pgn", type="selfplay_pgn")
                artifact.add_file(str(self.current_chunk_path.absolute()))
                wandb.log_artifact(artifact)

            self.new_chunk()


if __name__ == "__main__":
    replay_buffer = ReplayBuffer(
        workdir=Path() / "replay_buffer",
        pgn_chunk_games_size=10_000,
        window_games_size=500_000,
        batch_size=1,
        upload_pgn=False,
    )

    for _ in range(25):
        replay_buffer.add_game(
            [chess.Move.from_uci(move) for move in "f2f4 e7e5 g2g4 d8h4".split()]
        )

    datamodule = replay_buffer.prepare_dataset()

    for _ in range(25):
        replay_buffer.add_game(
            [chess.Move.from_uci(move) for move in "f2f4 e7e5 g2g4 d8h4".split()]
        )

    datamodule = replay_buffer.prepare_dataset()
