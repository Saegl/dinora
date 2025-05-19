"""
Fit a function of the form:
    cp(q) = a * q + b * q**3

This function maps MCTS evaluation values (`q`) to Stockfish-like centipawn scores (`cp`).

Why this form?
- `q` values come from the MCTS search and lie in the range [-1, 1]
- Stockfish scores (`cp`) are in centipawns and typically range from -1000 to +1000,
  with the most interesting positions often falling between -500 and +500
- We need this conversion to align MCTS evaluations with conventional engine output

Why this function?
- It's a simple 3rd-degree polynomial, easy to fit and evaluate
- It’s **odd**, satisfying:
    cp(0) = 0
    cp(x) = -cp(-x)

These symmetry properties match the evaluation domain, where a `q` of `+x` for White should
correspond to the opposite score `-x` for Black.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

npf64 = npt.NDArray[np.float64]

arr = np.load("qcp.npy")
q_values = arr[:, 0]
cp_values = arr[:, 1]

X = np.stack([q_values**i for i in [1, 3]], axis=1)

coeffs, *_ = np.linalg.lstsq(X, cp_values, rcond=None)
print("Coefficients:", coeffs)


def cp_from_q(q: npf64) -> npf64:
    a: float = coeffs[0]
    b: float = coeffs[1]
    return a * q + b * q**3


# Plotting
q_plot = np.linspace(-1, 1, 300)
cp_predicted = cp_from_q(q_plot)

plt.figure(figsize=(8, 6))
plt.scatter(q_values, cp_values, label="Data", color="blue", s=20)
plt.plot(q_plot, cp_predicted, label="Fitted odd polynomial", color="red", linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.xlabel("q value (in [-1, 1])")
plt.ylabel("Centipawn (cp)")
plt.title("Fitting Odd Polynomial to MCTS Q → Stockfish cp")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fit.png")
