import chess

from dinora.models.registry import ModelConfig, build_model
from dinora.search.mcts_batch.mcts_batch import (
    MctsParams,
    Node,
    collect_batch,
    expand,
    process_batch,
)

evaluator = build_model(ModelConfig("handcrafted"))
DEFAULT_PARAMS = MctsParams()


def check_no_virtual_visits(node: Node):
    assert node.virtual_visits == 0
    for child in node.children.values():
        check_no_virtual_visits(child)


def check_no_duplicate_leaves(leaves: list[Node]):
    ids = [id(leaf) for leaf in leaves]
    assert len(ids) == len(set(ids))


def test_collect_batch():
    board = chess.Board()
    priors, value = evaluator.evaluate(board)
    root = Node(None, -value, 1.0, chess.Move.null())
    expand(root, priors, DEFAULT_PARAMS.fpu)

    batch_boards, batch_leaves = collect_batch(root, board, 3.0, 16, 1, 1)

    assert len(batch_leaves) == len(batch_boards), "Sanity check"
    check_no_duplicate_leaves(batch_leaves)


def test_process_batch():
    board = chess.Board()
    priors, value = evaluator.evaluate(board)
    root = Node(None, -value, 1.0, chess.Move.null())
    expand(root, priors, DEFAULT_PARAMS.fpu)

    batch_boards, batch_leaves = collect_batch(root, board, 3.0, 16, 1, 1)
    process_batch(batch_boards, batch_leaves, evaluator, DEFAULT_PARAMS.fpu)
    check_no_virtual_visits(root)
