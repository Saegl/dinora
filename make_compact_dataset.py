"""
To run you need this deps:
pip install -e . chess numpy wandb requests tqdm
"""
import json
import pathlib
import concurrent.futures

import numpy as np

from dinora.pgntools import load_compact_state_tensors


def convert_dir(
    pgns_dir: pathlib.Path, save_dir: pathlib.Path, files_count: int | None
):
    pgns_dir = pgns_dir.absolute()
    save_dir = save_dir.absolute()

    tasks = [
        (path, save_dir / (path.name + "train.npz"))
        for path in pgns_dir.rglob("*.pgn")
        if path.is_file()
    ]

    run_parallel_convert(tasks[:files_count], save_dir)


def run_parallel_convert(
    tasks: list[tuple[pathlib.Path, pathlib.Path]], save_dir: pathlib.Path
):
    report = {
        "train": {},
        "test": {},
    }
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_pgn_file, pgn, save) for pgn, save in tasks]
        for future in concurrent.futures.as_completed(futures):
            save_path, states_count = future.result()
            print(f"File is converted: {save_path.name}, states {states_count}")
            report["train"][save_path.name] = states_count

    with open(save_dir / "report.json", "w") as f:
        json.dump(report, f)


def convert_pgn_file(pgn_path: pathlib.Path, save_path: pathlib.Path):
    boards = []
    policies = []
    outcomes = []

    print("Converting", pgn_path.name)

    with open(pgn_path, "r", encoding="utf8", errors="ignore") as pgn:
        for compact_state, (policy, outcome) in load_compact_state_tensors(pgn):
            boards.append(compact_state)
            policies.append(policy)
            outcomes.append(outcome)

    boards_np = np.array(boards, dtype=np.int64)
    policies_np = np.array(policies, dtype=np.int64)
    outcomes_np = np.array(outcomes, dtype=np.int64)

    np.savez_compressed(
        save_path, boards=boards_np, policies=policies_np, outcomes=outcomes_np
    )
    return save_path, len(boards_np)


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
        help="number of files to convert",
        type=int,
    )

    args = parser.parse_args()

    convert_dir(args.pgn_dir, args.output_dir, args.files_count)
