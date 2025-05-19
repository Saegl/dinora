"""
Collect (q, cp) pairs, where q is the MCTS evaluation and cp is stockfish centipawns value.

Used to fit the cp(q) function later.
"""

import chess
import numpy as np

from dinora.engine import Engine
from dinora.search.mcts.mcts import Node
from dinora.search.stoppers import NodesCount
from train.handmade_val_dataset.dataset import POSITIONS

engine = Engine(searcher="mcts")
engine.load_model()

array: list[list[float]] = []

for pos in POSITIONS:
    board = chess.Board(fen=pos["fen"])
    node = engine.searcher.search(board, NodesCount(300), engine.model)

    # TODO: remove hack, expand `Engine`
    assert isinstance(node, Node), (
        "Unfortunately, `Engine` doesn't return a `Node` by default."
        " You'll need to temporarily hack it to `return root`."
    )
    q = node.value_sum / node.visits
    v = -q

    cp = 100 * pos["stockfish_cp"]  # It wasn't actually in centipawns; multiply by 100
    is_black = pos["type"].startswith("BLACK")
    if is_black:  # Convert to relative perspective
        cp = -cp

    same_sign = (v < 0 and cp < 0) or (v > 0 and cp > 0)

    if abs(cp) > 50 * 100 or not same_sign:
        continue

    print("Model", v, "Stockfish", cp)
    array.append([v, cp])


out = np.array(array)
print(f"Data saved to `qcp.npy`, shape: {out.shape}")
np.save("qcp.npy", out)
