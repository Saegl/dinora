# type: ignore
# This file in broken state
import numpy as np

from dinora import PROJECT_ROOT
from dinora.encoders.board_representation import (
    compact_state_to_board_tensor,
    PLANE_NAMES,
)
from dinora.encoders.policy import index_to_move


SHORTER_NAMES = [
    name.replace("WHITE", "W").replace("BLACK", "B") for name in PLANE_NAMES
]


def render_board_state(board_state):
    bpr = 6  # boards per row
    grid_rows_count = len(PLANE_NAMES) // bpr + (len(PLANE_NAMES) % bpr != 0)
    bm = 3  # boards margin

    for grid_row in range(grid_rows_count):
        labels = SHORTER_NAMES[0 + bpr * grid_row : bpr + bpr * grid_row]
        labels = [label[0:10].ljust(10) for label in labels]
        print((" " * bm).join(labels))
        print((" " * bm).join(["----------"] * bpr))
        for row_idx in range(8):
            for rel_plane_idx in range(bpr):
                print("|", end="")
                for column_idx in range(8):
                    plane_idx = rel_plane_idx + bpr * grid_row
                    if plane_idx >= len(PLANE_NAMES):
                        square = 0.0
                    elif plane_idx == 16:
                        # TODO: Implement non binary cells in console renderer
                        # FIFTY MOVES is the only non binary cell
                        square = 0.0  # just ignore for now
                    else:
                        square = board_state[plane_idx][7 - row_idx][column_idx]
                    cell = "■" if square == 1.0 else " "
                    print(cell, end="")
                print("|", end="")
                print(" " * bm, end="")
            print()
        print((" " * bm).join(["----------"] * bpr))


def load_from_compact_dataset():
    data = np.load(PROJECT_ROOT / "data/converted_dataset/1.pgntrain.npz")
    boards = data["boards"]
    policies = data["policies"]
    outcomes = data["outcomes"]
    return enumerate(zip(boards, policies, outcomes))


def load_from_pgn_string():
    import io
    from dinora.pgntools import load_compact_state_tensors

    pgn_string = """
[Result "1-0"]

1. e4 f5 2. e5 d5 3. exd6 Bd7 4. dxc7 g5 5. Qh5#
    """

    handler = io.StringIO(pgn_string)
    for idx, (board, (policy, outcome)) in enumerate(
        load_compact_state_tensors(handler)
    ):
        yield idx, (board, policy, outcome)


if __name__ == "__main__":
    for state_idx, (compact_state, policy, outcome) in load_from_pgn_string():
        # for state_idx, (compact_state, policy, outcome) in load_from_compact_dataset():
        board_state = compact_state_to_board_tensor(compact_state)
        render_board_state(board_state)
        print("STATE IDX =", state_idx)
        print(
            "POLICY RAW =",
            policy,
            "DECODED =",
            index_to_move(policy, False),
            "DECODED (FLIPPED) =",
            index_to_move(policy, True),
        )
        print(
            "OUTCOME RAW =",
            outcome,
            "MEANS =",
            ["WE WIN", "DRAW", "WE LOSS"][int(outcome)],
        )
        try:
            input("next>")
        except KeyboardInterrupt:
            quit(0)
