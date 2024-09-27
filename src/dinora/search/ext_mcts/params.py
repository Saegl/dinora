# If you do `from __future__ import annotations` here
# you will break engine.py

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from dinora.search.ext_mcts.node import Node


def init_puct(params: "MCTSparams", node: Node) -> Node:
    return node.best_puct(params.cpuct)


def init_softmax(params: "MCTSparams", node: Node) -> Node:
    return node.best_softmax(params.t)


selection_policies = {
    "puct": init_puct,
    "softmax": init_softmax,
}


@dataclass
class MCTSparams:
    # Reduce node to terminals with MCTS solver
    # NOTE: currently unstable, may misevaluate states
    node_reduction: bool = field(default=False)

    # First Play Urgency - value of unvisited nodes
    fpu: float = field(default=-1.0)
    fpu_at_root: float = field(default=0.0)

    # exploration parameter
    selection_policy_name: Literal["puct", "softmax"] = field(default="puct")
    cpuct: float = field(default=3.0)
    t: float = field(default=1.0)

    # random
    dirichlet_alpha: float = field(default=0.3)
    noise_eps: float = field(default=0.0)  # set to 0.0 to disable random

    send_func: Callable[[str], None] = print

    def selection_policy(self, node: Node) -> Node:
        policy = selection_policies[self.selection_policy_name]
        return policy(self, node)
