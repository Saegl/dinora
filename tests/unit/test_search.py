import chess
from dinora.mcts import run_mcts, NodesCountConstraint, MCTSparams
from dinora.mcts.uci_info import cp


# from dinora.models.handcrafted import DummyModel as TestModel  # Fast model

from dinora.models.dnn import DNNModel as TestModel  # Real model

model = TestModel()


def test_cp():
    # Dead draw
    assert cp(0.0) == 0

    # q = 1.0 absolute win
    # q = -1.0 absolute lose
    # and they opposite to each other
    assert cp(1.0) == -cp(-1.0)


def test_search_starting_position():
    board = chess.Board()
    nodes_count = 20
    root = run_mcts(
        board,
        constraint=NodesCountConstraint(nodes_count),
        evaluator=model,
        params=MCTSparams,
    )
    assert root.children
    assert root.is_root()

    assert root.number_visits == nodes_count

    bestnode = root.get_most_visited_node()
    bestmove = bestnode.move

    assert bestmove is not None
    assert bestmove.uci() in {"e2e4", "d2d4", "c2c4", "g1f3"}

    # White is slightly better at starting position
    # assert 0.0 <= bestnode.Q() <= 0.5


# TODO: test draw? mate in 1, 2?
