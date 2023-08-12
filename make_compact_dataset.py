"""
Steps to make your own dataset from pgn games:
1. Download your pgn to data/<folder_name>/<your.pgn>
2. Split your pgn to smaller parts of 10k games each with 
`pgn-extract.exe -#10000 <your.pgn>` and place it under
data/<folder_name2>
3. Install deps for this script
`pip install -e . chess numpy wandb requests tqdm`
4. Run this program with
`make_compact_dataset.py <folder_name2> <folder_name3>`
You can append `--q-nodes=10` if you want add stockfish evals to
the resulting dataset
5. Now you can train neural network with this dataset, use <folder_name3>
as `dataset_label` in dinora.train
"""
import json
import logging
import pathlib
import concurrent.futures

import chess.engine
from chess.engine import LOGGER as engine_logger

import numpy as np

from dinora.board_representation import board_to_compact_state
from dinora.pgntools import load_game_states
from dinora.policy import policy_index
from dinora.outcome import wdl_index, z_value, stockfish_value


def convert_dir(
    pgns_dir: pathlib.Path,
    save_dir: pathlib.Path,
    files_count: int | None,
    q_nodes: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
):
    if not (0.9 <= train_percentage + val_percentage + test_percentage <= 1.1):
        raise ValueError("Train/val/test percentage doesn't sum to 1.0")

    pgns_dir = pgns_dir.absolute()
    save_dir = save_dir.absolute()
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (path, save_dir / (path.name + "train.npz"), q_nodes)
        for path in pgns_dir.rglob("*.pgn")
        if path.is_file()
    ]

    chunks = run_parallel_convert(tasks[:files_count])
    generate_report(
        chunks,
        save_dir,
        train_percentage,
        val_percentage,
        test_percentage,
    )


def run_parallel_convert(
    tasks: list[tuple[pathlib.Path, pathlib.Path]]
) -> dict[str, int]:
    chunks: dict[str, int] = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_pgn_file, *args) for args in tasks]
        for future in concurrent.futures.as_completed(futures):
            save_path, states_count = future.result()
            print(f"File is converted: {save_path.name}, states {states_count}")
            chunks[save_path.name] = states_count
    return chunks


def generate_report(
    chunks: dict[str, int],
    save_dir: pathlib.Path,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
):
    total_states = sum(chunks.values())
    if not (0.9 <= train_percentage + val_percentage + test_percentage <= 1.1):
        raise ValueError("Train/val/test percentage doesn't sum to 1.0")

    train_sum = total_states * train_percentage
    val_sum = total_states * val_percentage

    train_chunks, val_chunks, test_chunks = {}, {}, {}
    train_current, val_current, test_current = 0, 0, 0

    for name, states in chunks.items():
        if train_current == 0 or train_current + states <= train_sum:
            train_chunks[name] = states
            train_current += states
        elif val_current == 0 or val_current + states <= val_sum:
            val_chunks[name] = states
            val_current += states
        else:
            test_chunks[name] = states
            test_current += states

    report = {
        "train": train_chunks,
        "val": val_chunks,
        "test": test_chunks,
        "meta": {
            "train_percentage": train_current / total_states,
            "val_percentage": val_current / total_states,
            "test_percentage": test_current / total_states,
        },
    }

    with open(save_dir / "report.json", "w") as f:
        json.dump(report, f)


def convert_pgn_file(pgn_path: pathlib.Path, save_path: pathlib.Path, q_nodes: int):
    print("Converting", pgn_path.name)

    tensors = {name: [] for name in ["boards", "policies", "wdls", "z_values"]}

    if q_nodes > 0:
        engine_logger.setLevel(logging.ERROR)
        # TODO: remove this debug msg = DEBUG:asyncio:Using proactor: IocpProactor
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure({"UCI_ShowWDL": True})
        tensors["q_values"] = []

    try:
        with open(pgn_path, "r", encoding="utf8", errors="ignore") as pgn:
            for game, board, move in load_game_states(pgn):
                tensors["boards"].append(board_to_compact_state(board))
                tensors["policies"].append(policy_index(move, not board.turn))
                tensors["wdls"].append(wdl_index(game, board.turn))
                tensors["z_values"].append(z_value(game, board.turn))

                if q_nodes > 0:
                    tensors["q_values"].append(stockfish_value(board, engine, q_nodes))
    finally:
        if q_nodes > 0:
            engine.close()

    tensors["boards"] = np.array(tensors["boards"], dtype=np.int64)
    tensors["policies"] = np.array(tensors["policies"], dtype=np.int64)
    tensors["wdls"] = np.array(tensors["wdls"], dtype=np.int64)
    tensors["z_values"] = np.array(tensors["z_values"], dtype=np.float32).reshape(-1, 1)

    if q_nodes > 0:
        tensors["q_values"] = np.array(tensors["q_values"], dtype=np.float32).reshape(
            -1, 1
        )

    np.savez_compressed(save_path, **tensors)
    return save_path, len(tensors["boards"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool to convert pgns to compact dataset",
    )

    parser.add_argument(
        "pgn_dir",
        help="directory of pgn files",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_dir",
        help="directory to save compact dataset",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--files-count",
        help="number of files to convert, all if not specified",
        type=int,
    )
    parser.add_argument(
        "--q-nodes",
        help="pass positive integer to add stockfish q values",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train",
        help="train percentage",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val",
        help="val percentage",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test",
        help="test percentage",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    convert_dir(
        args.pgn_dir,
        args.output_dir,
        args.files_count,
        args.q_nodes,
        args.train,
        args.val,
        args.test,
    )
