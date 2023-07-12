"""
To run you need this deps:
pip install -e . chess numpy wandb requests tqdm
"""
import json
import concurrent.futures
from pathlib import Path

import numpy as np

from dinora.pgntools import load_compact_state_tensors
from dinora.ccrl import download_ccrl_dataset


PROJECT_DIR = Path(__file__).parent


def convert_ccrl_to_bin_dataset(
    chunks_count: int
):
    train_paths, test_paths = download_ccrl_dataset(
        chunks_count=chunks_count,
    )

    save_dir = PROJECT_DIR / "data/converted_dataset"
    save_dir.mkdir(parents=True, exist_ok=True)

    # list[(pgn_path, save_path)]
    tasks = []
    tasks += [(path, save_dir / (path.name + "train.npz")) for path in train_paths]
    tasks += [(path, save_dir / (path.name + "test.npz")) for path in test_paths]

    report = {
        'train': {},
        'test': {},
    }
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert, pgn, save) for pgn, save in tasks]
        for future in concurrent.futures.as_completed(futures):
            save_path, states_count = future.result()
            print(f"File is converted: {save_path}, states {states_count}")

            rel_path = save_path.relative_to(save_dir)
            rel_path_str = str(rel_path)

            label = 'train' if 'train' in rel_path_str else 'test'
            report[label][rel_path_str] = states_count
    
    with open(save_dir / 'report.json', 'w') as f:
        json.dump(report, f)



def convert(pgn_path: Path, save_path: Path):
    boards = []
    policies = []
    outcomes = []

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
        description="Tool to convert ccrl pgns to compact dataset"
    )
    parser.add_argument(
        "--chunks_count",
        help="Number of chunks (max 250)",
        type=int,
        default=250,
    )
    args = parser.parse_args()

    convert_ccrl_to_bin_dataset(
        chunks_count=args.chunks_count,
    )
