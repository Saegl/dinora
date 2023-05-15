import concurrent.futures
from pathlib import Path

import wandb
import numpy as np

from dinora.board_representation2 import board_to_compact_state
from dinora.preprocess_pgn import load_game_states, outcome_tensor
from dinora.dataset import download_ccrl_dataset
from dinora.policy2 import policy_index_tensor


PROJECT_DIR = Path(__file__).parent


def convert_ccrl_to_bin_dataset(
    chunks_count: int, upload: bool, delete_after_upload: bool
):
    artifact = None
    if upload:
        run = wandb.init(
            project="dinora-chess",
            job_type="upload_dataset",
        )

        artifact = wandb.Artifact(
            name="ccrl-dataset",
            type="dataset",
        )

    train_paths, test_paths = download_ccrl_dataset(
        chunks_count=chunks_count,
    )

    save_dir = PROJECT_DIR / "data/converted_dataset"
    save_dir.mkdir(parents=True, exist_ok=True)

    # list[(pgn_path, save_path)]
    tasks = []
    tasks += [(path, save_dir / (path.name + "train.npz")) for path in train_paths]
    tasks += [(path, save_dir / (path.name + "test.npz")) for path in test_paths]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_save_path = {executor.submit(convert, p[0], p[1]): p[1] for p in tasks}
        for future in concurrent.futures.as_completed(future_to_save_path):
            save_path = future_to_save_path[future]
            print("File is converted: ", str(save_path))

            if artifact:
                print(str(save_path))
                artifact.add_file(str(save_path))

            if delete_after_upload:
                save_path.unlink()

    if artifact:
        run.log_artifact(artifact)


def convert(pgn_path: Path, save_path: Path):
    boards = []
    policies = []
    outcomes = []

    for game, board, move in load_game_states(pgn_path):
        flip = not board.turn
        boards.append(board_to_compact_state(board))
        policies.append(policy_index_tensor(move, flip))
        outcomes.append(outcome_tensor(game, flip))

    boards_np = np.array(boards, dtype=np.int64)
    policies_np = np.array(policies, dtype=np.int64)
    outcomes_np = np.array(outcomes, dtype=np.int64)

    np.savez_compressed(
        save_path, boards=boards_np, policies=policies_np, outcomes=outcomes_np
    )


if __name__ == "__main__":
    convert_ccrl_to_bin_dataset(
        chunks_count=250,
        upload=True,
        delete_after_upload=True,
    )
