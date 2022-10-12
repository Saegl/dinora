import json
import pytest
import chess
import numpy as np

from dinora.board_representation import canon_input_planes

with open("tests/data/positions/various.json") as f:
    various = json.load(f)

canon_planes = np.load("tests/data/positions/canon_planes.npy")


@pytest.mark.parametrize("index, pos", zip(range(len(various)), various))
def test_bench_canon_input_planes(index, pos, benchmark):
    """
    This test checks the speed of converting board into planes
    If you are willing to update canon_input_planes representation,
    look at bin/update_canon_tests
    """
    board = chess.Board(pos)
    planes = benchmark(canon_input_planes, pos, not board.turn)

    assert planes.shape == (18, 8, 8)
    assert np.allclose(planes, canon_planes[index])
