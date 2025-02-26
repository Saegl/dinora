from pathlib import Path

import numpy as np

from dataset.encoders.compact_board_tensor import compact_state_to_board_tensor
from dataset.make import convert_pgn_file
from dinora.encoders.board_tensor import board_to_tensor
from dinora.pgntools import load_game_states


def test_compact_boards_load(tmp_path: Path):
    pgn_file = Path("tests/data/pgns/example.pgn")
    save_path = tmp_path / "example.npz"

    converted_path, states_count = convert_pgn_file(pgn_file, save_path, 0)
    assert states_count > 0

    tensors = np.load(converted_path)

    with pgn_file.open() as f:
        for cb, (_, board, _) in zip(tensors["boards"], load_game_states(f)):
            loaded_board = compact_state_to_board_tensor(cb)
            direct_board = board_to_tensor(board, not board.turn)
            assert np.allclose(loaded_board, direct_board)
