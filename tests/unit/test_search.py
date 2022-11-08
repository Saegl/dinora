from dinora.mcts.uci_info import cp


def test_cp():
    # Dead draw
    assert cp(0.0) == 0

    # q = 1.0 absolute win
    # q = -1.0 absolute lose
    # and they opposite to each other
    assert cp(1.0) == -cp(-1.0)
