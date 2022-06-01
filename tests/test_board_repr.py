import json
import chess
import pytest
import numpy as np

from dinora.board_representation.canon_planes import canon_input_planes

with open("tests/positions/various.json") as f:
    various = json.load(f)

canon_planes = np.load("tests/positions/canon_planes.npy")


@pytest.mark.parametrize("index, pos", zip(range(len(various)), various))
def test_canon_input_not_changed(index, pos):
    """
    This test checks you didn't change the canon planes representation.
    Useful if you changed canon_input_planes to gain performance,
    but accidentally broke the canon planes representation.
    """
    board = chess.Board(pos)
    planes = canon_input_planes(pos, not board.turn)

    assert planes.shape == (18, 8, 8)
    assert np.allclose(planes, canon_planes[index])
