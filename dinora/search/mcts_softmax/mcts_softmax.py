from __future__ import annotations

import math
import random

import chess

from dinora.models import BaseModel
from dinora.models.base import Priors
from dinora.search.base import BaseSearcher, ConfigType, DefaultValue
from dinora.search.stoppers import MoveTime, NodesCount, Stopper

RANDOM_FACTOR = 0.3
MAX_BATCH_SIZE = 512
MAX_COLLISION_RATE = 0.5


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


def expand(node: Node, priors: Priors) -> None:
    for move, prior in priors.items():
        node.children[move] = Node(node, prior)


def backpropagate(node: Node, value: float) -> None:
    current = node
    turn_factor = -1
    while current.parent:
        current.visits += 1
        current.value_sum += turn_factor * value
        current = current.parent
        turn_factor *= -1

    current.visits += 1


def select_leaf(root: Node, board: chess.Board) -> Node:
    current = root

    while current.is_expanded():
        move, current = select_child(current)
        board.push(move)

    return current


def select_child(node: Node) -> tuple[chess.Move, Node]:
    return max(node.children.items(), key=lambda el: ucb_score(node, el[1]))


def ucb_score(parent: Node, child: Node) -> float:
    exploration = 1.25 * child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    return child.value() + exploration + RANDOM_FACTOR * (random.random() - 0.5)


def most_visits_move(root: Node) -> chess.Move:
    bestmove, _ = max(root.children.items(), key=lambda el: el[1].visits)
    return bestmove


class MCTSSoftmax(BaseSearcher):
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
        priors, value = evaluator.evaluate(board)
        expand(root, priors)

        total_collisions = 0
        total_non_collisions = 0

        iterations = 0

        total_batch_nodes = 0
        max_batch_nodes = 0

        while not stopper.should_stop():
            leafs = []
            boards = []
            collisions = 0
            non_collisions = 0

            for _ in range(MAX_BATCH_SIZE):
                leaf_board = board.copy()
                leaf = select_leaf(root, leaf_board)
                if not leaf.in_batch:
                    non_collisions += 1

                    leaf.in_batch = True
                    leafs.append(leaf)
                    boards.append(leaf_board)
                else:
                    collisions += 1

                    colrate = collisions / (collisions + non_collisions)
                    if colrate > MAX_COLLISION_RATE:
                        break

            for leaf, (priors, value) in zip(leafs, evaluator.evaluate_batch(boards)):
                leaf.in_batch = False
                expand(leaf, priors)
                backpropagate(leaf, value)

            total_batch_nodes += len(boards)
            max_batch_nodes = max(max_batch_nodes, len(boards))

            total_collisions += collisions
            total_non_collisions += non_collisions

            iterations += 1

        col_rate = total_collisions / (total_collisions + total_non_collisions)
        print(f"info nodes {root.visits}")
        # print(f"Collisions rate: {col_rate:.2f}")
        # print(f"iterations {iterations}")
        # print(f"batch_size {total_batch_nodes / iterations}")
        # print(f"max batch nodes {max_batch_nodes}")
        # print()
        return most_visits_move(root)


def selfplay(searcher: MCTSSoftmax, model: BaseModel) -> None:
    board = chess.Board()
    while not board.outcome(claim_draw=True):
        move = searcher.search(board, MoveTime(5_000), model)
        board.push(move)


if __name__ == "__main__":
    from dinora.models import model_selector

    model = model_selector("alphanet", None, "cuda")
    searcher = MCTSSoftmax()

    print("Expected nodes:", 100 * MAX_BATCH_SIZE)
    # searcher.search(chess.Board(), NodesCount(100), model)
    selfplay(searcher, model)
