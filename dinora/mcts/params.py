from dataclasses import dataclass, field
from dinora.cli.uci_options import FloatString


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
    c: float = field(default=2.0, metadata={"uci_option_type": FloatString()})

    # random
    dirichlet_alpha: float = field(
        default=0.3, metadata={"uci_option_type": FloatString()}
    )
    noise_eps: float = field(
        default=0.0, metadata={"uci_option_type": FloatString()}
    )  # set to 0.0 to disable random
