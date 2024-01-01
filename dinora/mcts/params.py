from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class MCTSparams:
    # Reduce node to terminals with MCTS solver
    # NOTE: currently unstable, may misevaluate states
    node_reduction: bool = field(default=False)

    # First Play Urgency - value of unvisited nodes
    fpu: float = field(default=-1.0)
    fpu_at_root: float = field(default=0.0)

    # exploration parameter
    cpuct: float = field(default=3.0)

    # random
    dirichlet_alpha: float = field(default=0.3)
    noise_eps: float = field(default=0.0)  # set to 0.0 to disable random

    send_func: Callable[[str], None] = print
