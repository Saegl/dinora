"""
Noise can be applied at root
to change engine behavior in each game
"""
import numpy as np
from dinora.models.base import Priors


def apply_noise(priors: Priors, dirichlet_alpha: float, noise_eps: float) -> Priors:
    noise = np.random.dirichlet([dirichlet_alpha] * len(priors))
    return {
        move: (1 - noise_eps) * prior + noise_eps * noise[i]
        for i, (move, prior) in enumerate(priors.items())
    }
