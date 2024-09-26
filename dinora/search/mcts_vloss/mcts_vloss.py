from __future__ import annotations

import math
import time

import chess

from dinora.models import BaseModel
from dinora.search.base import BaseSearcher, ConfigType, DefaultValue
from dinora.search.stoppers import NodesCount, Stopper

VIRTUAL_LOSS = 20
BATCH_SIZE = 128


class Node:
    parent: Node | None
    visits: int
    prior: float
    children: dict[chess.Move, Node]
    in_batch: bool

    def __init__(self, parent: Node | None, prior: float) -> None:
        self.parent = parent
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.in_batch = False

    def is_expanded(self) -> bool:
        return len(self.children) != 0

    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def expand(node: Node, priors):
    for move, prior in priors.items():
        node.children[move] = Node(node, prior)


def backpropagate(node: Node, value: float) -> None:
    current = node
    turn_factor = -1
    while current.parent:
        current.visits -= VIRTUAL_LOSS - 1
        current.value_sum += turn_factor * value
        current = current.parent
        turn_factor *= -1

    current.visits -= VIRTUAL_LOSS - 1

def undo_virtual_loss(node: Node) -> None:
    current = node
    current.visits -= VIRTUAL_LOSS
    while current.parent:
        current = current.parent
        current.visits -= VIRTUAL_LOSS


def select_leaf(root: Node, board: chess.Board) -> Node:
    current = root
    current.visits += VIRTUAL_LOSS

    while current.is_expanded():
        move, current = select_child(current)
        current.visits += VIRTUAL_LOSS
        board.push(move)

    return current


def select_child(node: Node) -> tuple[chess.Move, Node]:
    return max(node.children.items(), key=lambda el: ucb_score(node, el[1]))


def ucb_score(parent: Node, child: Node) -> float:
    exploration = 1.25 * child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    return child.value() + exploration


def most_visits_move(root: Node) -> chess.Move:
    bestmove, _ = max(root.children.items(), key=lambda el: el[1].visits)
    return bestmove


class MCTSVloss(BaseSearcher):
    def __init__(self) -> None:
        pass

    def config_schema(self) -> dict[str, tuple[ConfigType, DefaultValue]]:
        return {}

    def set_config_param(self, k: str, v: str) -> None:
        return None

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        root = Node(None, 0.0)
        root_board = board
        priors, value = evaluator.evaluate(root_board)
        expand(root, priors)
        while not stopper.should_stop():
            leafs = []
            leaf_boards = []
            for _ in range(BATCH_SIZE):
                leaf_board = root_board.copy()
                leaf = select_leaf(root, leaf_board)
                if not leaf.in_batch:
                    leaf.in_batch = True
                    leafs.append(leaf)
                    leaf_boards.append(leaf_board)
                else:
                    undo_virtual_loss(leaf)

            print(len(leafs))

            evals = evaluator.evaluate_batch(leaf_boards)

            for leaf, (priors, value) in zip(leafs, evals):
                leaf.in_batch = False
                expand(leaf, priors)
                backpropagate(leaf, value)


        best_line = []
        current = root
        while current.children:
            move, current = select_child(current)
            best_line.append(move.uci())

        print("Best line", ", ".join(best_line))
        print(f"info nodes {root.visits}")
        return most_visits_move(root)


if __name__ == "__main__":
    # from dinora.models.onnxmodel import OnnxModel
    from dinora.models import model_selector

    # model = OnnxModel(device="cuda")
    model = model_selector("alphanet", None, "cuda")


    searcher = MCTSVloss()

    start_time = time.time()

    searcher.search(chess.Board(), NodesCount(30), model)

    end_time = time.time()

    print(f"Time taken {end_time - start_time}")

