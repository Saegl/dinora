from dataclasses import dataclass


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
