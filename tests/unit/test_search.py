from dinora import search


def test_cp():
    # Dead draw
    assert search.cp(0.0) == 0

    # q = 1.0 absolute win
    # q = -1.0 absolute lose
    # and they opposite to each other
    assert search.cp(1.0) == -search.cp(-1.0)
