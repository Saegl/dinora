# If you do `from __future__ import annotations` here
# you will break engine.py

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from dinora.options import param
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
    fpu: float = param(
        default=-1.0,
        minimum=-10.0,
        maximum=10.0,
        doc="First Play Urgency, the value assumed for a move before it has been "
        "searched. Higher values make the engine optimistic about untried moves, "
        "which helps it look for an escape in a bad position.",
    )
    fpu_at_root: float = param(
        default=0.0,
        minimum=-10.0,
        maximum=10.0,
        doc="Value the root node itself starts with, before its own evaluation is "
        "backed up. Moves are still seeded with `fpu`.",
    )

    # exploration parameter
    selection_policy_name: Literal["puct", "softmax"] = field(default="puct")
    cpuct: float = param(
        default=3.0,
        minimum=0.0,
        maximum=20.0,
        doc="Exploration constant. Higher values spread the search over more moves, "
        "lower values dig into the few moves the network already likes.",
    )
    t: float = param(
        default=1.0,
        minimum=0.01,
        maximum=10.0,
        doc="Softmax temperature, only used when `selection_policy_name` is softmax. "
        "Higher values make the choice between moves more uniform.",
    )

    # random
    dirichlet_alpha: float = param(
        default=0.3,
        minimum=0.01,
        maximum=10.0,
        doc="Shape of the Dirichlet noise. Lower values pile the noise onto a few "
        "moves, higher values spread it evenly. Has no effect if `noise_eps` is 0.",
    )
    # set noise_eps to 0.0 to disable random
    noise_eps: float = param(
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        doc="How much random noise is blended into the network policy at the root. "
        "Adds variety to the games the engine plays, this searcher applies it on "
        "every move, not only in the opening.",
    )

    send_func: Callable[[str], None] = print

    def selection_policy(self, node: Node) -> Node:
        policy = selection_policies[self.selection_policy_name]
        return policy(self, node)
