import chess
from io import StringIO
from chess.pgn import read_game
from dinora.policy import (
    INDEX_TO_MOVE,
    INDEX_TO_FLIPPED_MOVE,
    policy_index,
    extract_prob_from_policy,
)


def test_no_duplicates():
    assert len(INDEX_TO_MOVE) == len(set(INDEX_TO_MOVE))


def test_no_duplicates_in_flipped_moves():
    assert len(INDEX_TO_FLIPPED_MOVE) == len(set(INDEX_TO_FLIPPED_MOVE))


def test_contain_all_moves():
    """
    Of course it's hard to check every possible move by one pgn game
    but this crazy game contain all type of promotions, en passant
    and some simple vertical, horizontal and diagonal moves
    """

    all_moves_set = set(INDEX_TO_MOVE)
    flipped_moves_set = set(INDEX_TO_FLIPPED_MOVE)
    crazy_game = """
    [Event "?"]
    [Site "?"]
    [Date "2022.11.20"]
    [Round "?"]
    [White "Saegl"]
    [Black "Saegl"]
    [Result "*"]
    [ECO "D30"]
    [Opening "Queen's gambit declined"]
    [Termination "unterminated"]
    [TimeControl "inf"]

    1. d4 d5 {1.1s} 2. c4 {4.3s} e6 {0.83s} 3. c5 {0.64s} f6 {0.66s} 4. c6
    g6 {0.62s} 5. cxb7 {0.60s} g5 {0.76s} 6. bxa8=N {4.0s} h5 {236s} 7. b4 {0.97s}
    h4 {0.79s} 8. b5 {0.69s} Rh7 {0.83s} 9. b6 {0.61s} Rh8 {0.65s} 10. bxc7 {0.70s}
    Rh7 {0.68s} 11. cxd8=B {2.1s} Rh8 {1.5s} 12. e4 {1.6s} Rh7 {0.78s}
    13. exd5 {0.68s} Rh8 {0.69s} 14. dxe6 {0.62s} Rh7 {0.63s} 15. e7 {0.55s}
    Rh6 {0.84s} 16. exf8=Q+ {2.1s} Kxf8 {1.2s} 17. f4 {3.0s} Rh7 {0.75s}
    18. fxg5 {0.66s} Rh6 {0.52s} 19. gxh6 {0.52s} Ne7 {1.0s} 20. h7 {0.66s}
    Nf5 {0.90s} 21. h8=R+ {2.7s} Kg7 {2057s} 22. g4 {1.0s} hxg3 {1.2s} *
    """

    game = read_game(StringIO(crazy_game))
    for move in game.mainline_moves():
        assert move.uci() in all_moves_set
        assert move.uci() in flipped_moves_set


def test_flipped_and_not_flipped():
    move = chess.Move.from_uci("e2e4")
    assert policy_index(move, False) != policy_index(move, True)


def test_flipped_and_not_flipped_symmetry():
    move1 = chess.Move.from_uci("e2e4")
    move2 = chess.Move.from_uci("e7e5")

    assert policy_index(move1, True) == policy_index(move2, False)
