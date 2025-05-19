import math
from dataclasses import dataclass, field
from typing import Optional

import chess

from dinora.models.base import BaseModel, Priors
from dinora.search.base import BaseSearcher
from dinora.search.logger import UCILogger
from dinora.search.noise import apply_noise
from dinora.search.stoppers import Stopper


@dataclass
class MctsParams:
    # exploration constant
    cpuct: float = field(default=3.0)
    fpu: float = field(default=-1.0)
    # random
    opening_noise_moves: int = field(default=15)
    dirichlet_alpha: float = field(default=0.3)
    noise_eps: float = field(default=0.0)  # set to 0.0 to disable random


class Node:
    parent: Optional["Node"]
    children: dict[chess.Move, "Node"]
    value_sum: float
    visits: int
    prior: float
    move: chess.Move

    def __init__(
        self, parent: Optional["Node"], value: float, prior: float, move: chess.Move
    ):
        self.parent = parent
        self.children = {}
        self.value_sum = value  # value_sum to the perspective of parent
        self.visits = 1
        self.prior = prior
        self.move = move

    def puct(self, cpuct: float) -> float:
        assert self.parent
        exploitation = self.value_sum / self.visits
        exploration = cpuct * math.sqrt(self.parent.visits) * self.prior / self.visits
        return exploitation + exploration

    def __repr__(self) -> str:
        return f"Node <{self.move} {self.visits} {self.value_sum}>"


def select_best_puct(node: Node, cpuct: float) -> Node:
    best = None
    max_puct = None

    for child in node.children.values():
        puct = child.puct(cpuct)
        if max_puct is None or puct > max_puct:
            max_puct = puct
            best = child

    assert best is not None
    return best


def select_leaf(root: Node, board: chess.Board, cpuct: float) -> Node:
    node = root
    while len(node.children) != 0:
        node = select_best_puct(node, cpuct)
        board.push(node.move)
    return node


def expand(node: Node, child_priors: Priors, fpu: float) -> None:
    for move, prior in child_priors.items():
        node.children[move] = Node(node, fpu, prior, move)


def backup(leaf: Node, board: chess.Board, value_leaf: float) -> None:
    node = leaf
    value = value_leaf
    while node.parent:
        value = -value  # perspective alternating on each move
        node.value_sum += value
        node.visits += 1

        board.pop()
        node = node.parent

    # at this point we don't have parent, means we root
    root = node
    value = -value
    root.value_sum += value
    root.visits += 1


def most_visited_move(node: Node) -> chess.Move:
    max_visits = 0
    current_move = chess.Move.null()
    for move, child in node.children.items():
        if child.visits > max_visits:
            max_visits = child.visits
            current_move = move
    assert current_move != chess.Move.null(), "Can't play null move"
    return current_move


def terminal_solver(board: chess.Board) -> float | None:
    board_result = board.result(claim_draw=True)
    if board_result != "*":  # This node is terminal
        # No matter White or Black won, we lost because it is our turn to move
        if board_result == "1-0" or board_result == "0-1":
            return -1.0
        elif board_result == "1/2-1/2":
            return 0.0
    return None


def get_pv(root: Node, cpuct: float, maxlen: int = 15) -> tuple[str, int]:
    """
    Returns principal variation (PV) as a UCI move string and its depth in plies.
    The PV may be truncated by `maxlen`, but the actual depth reached is still returned.
    """
    node = root
    moves: list[str] = []
    depth = 1

    while len(node.children) != 0:
        node = select_best_puct(node, cpuct)
        depth += 1
        if len(moves) < maxlen:
            moves.append(node.move.uci())

    pvline = " ".join(moves)
    return (pvline, depth)


def cp(q: float) -> int:
    """
    Converts MCTS `q` value to Stockfish-like centipawn score.

    See `src/train/cp` to learn how these coefficients were obtained.
    """
    a = 85.59159393
    b = 584.76371188
    return int(a * q + b * q**3)


class MCTS(BaseSearcher[MctsParams]):
    def __init__(self) -> None:
        self.params = MctsParams()

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        logger = UCILogger()

        priors, value = evaluator.evaluate(board)
        root = Node(None, value, 1.0, chess.Move.null())
        if board.ply() < 2 * self.params.opening_noise_moves:
            priors = apply_noise(
                priors,
                dirichlet_alpha=self.params.dirichlet_alpha,
                noise_eps=self.params.noise_eps,
            )

        expand(root, priors, self.params.fpu)

        while not stopper.should_stop():
            leaf = select_leaf(root, board, self.params.cpuct)
            terminal_value = terminal_solver(board)
            if terminal_value is not None:
                priors, value = {}, terminal_value
            else:
                priors, value = evaluator.evaluate(board)
            expand(leaf, priors, self.params.fpu)
            backup(leaf, board, value)

            if logger.should_log():
                q = root.value_sum / root.visits
                v = -q  # q is evaluation on child, not root itself
                pv, depth = get_pv(root, self.params.cpuct)
                logger.on_search_iter(nodes=root.visits, pv=pv, depth=depth, cp=cp(v))

        logger.on_search_finish(nodes=root.visits)
        return most_visited_move(root)
