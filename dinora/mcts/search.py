from dataclasses import dataclass

import chess

from dinora.mcts.node import Node
from dinora.mcts.uci_info import UciInfo
from dinora.mcts.noise import apply_noise
from dinora.mcts.constraints import Constraint
from dinora.mcts.evaluator import Evaluator, Priors


@dataclass
class MCTSparams:
    # First Play Urgency - value of unvisited nodes
    fpu: float = -1.0
    fpu_at_root: float = 0.0

    # exploration parameter
    c: float = 2.0

    # random
    dirichlet_alpha: float = 0.3
    noise_eps: float = 0.0  # set to 0.0 to disable random


def selection(root: Node, c: float) -> Node:
    current = root

    while current.is_expanded and current.children:
        current = current.best_child(c)

    return current


def expansion(node: Node, child_priors: Priors, fpu: float) -> None:
    node.is_expanded = True

    for move, prior in child_priors.items():
        node.add_child(move, prior, fpu)


def backpropagation(node: Node, value_estimate: float) -> None:
    current = node
    # Child nodes are multiplied by -1 because we want max(-opponent eval)
    turnfactor = -1
    while current.parent is not None:
        current.number_visits += 1
        current.total_value += value_estimate * turnfactor
        current = current.parent
        turnfactor *= -1
    current.number_visits += 1


def run_mcts(
    board: chess.Board,
    constraint: Constraint,
    evaluator: Evaluator,
    params: MCTSparams,
) -> Node:
    uci_info = UciInfo()

    root = Node(params.fpu_at_root, lazyboard=board)
    child_priors, value_estimate = evaluator.evaluate(root.board)
    child_priors = apply_noise(
        child_priors,
        params.dirichlet_alpha,
        params.noise_eps,
    )
    expansion(root, child_priors, params.fpu)
    backpropagation(root, value_estimate)

    while constraint.meet():
        leaf = selection(root, params.c)
        child_priors, value_estimate = evaluator.evaluate(root.board)
        expansion(leaf, child_priors, params.fpu)
        backpropagation(leaf, value_estimate)
        uci_info.after_iteration(root)

    uci_info.at_mcts_end(root)
    return root
