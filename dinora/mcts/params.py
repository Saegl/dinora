from typing import Callable
from dataclasses import dataclass, field
from dinora.uci.uci_options import FloatString  # TODO: uncouple uci and mcts


@dataclass
class MCTSparams:
    # First Play Urgency - value of unvisited nodes
    fpu: float = field(
        default=-1.0,
        metadata={
            "uci_option_type": FloatString(),
        },
    )
    fpu_at_root: float = field(default=0.0, metadata={"uci_option_type": FloatString()})

    # exploration parameter
    cpuct: float = field(default=3.0, metadata={"uci_option_type": FloatString()})

    # random
    dirichlet_alpha: float = field(
        default=0.3, metadata={"uci_option_type": FloatString()}
    )
    noise_eps: float = field(
        default=0.0, metadata={"uci_option_type": FloatString()}
    )  # set to 0.0 to disable random

    send_func: Callable[[str], None] = print
