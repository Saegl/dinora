from typing import Any
import chess
import numpy as np


def create_flipped_uci_labels() -> list[str]:
    """
    Seems to somehow transform the labels used for describing the universal chess interface format, putting
    them into a returned list.
    :return:
    """

    def repl(x: str) -> str:
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in create_uci_labels()]


def create_uci_labels() -> list[str]:
    """
    Creates the labels for the UCI into an array and returns them
    """
    labels_array = []
    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]  # horizontal
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8"]  # vertical
    promoted_to = ["q", "r", "b", "n"]

    # UCI move is like e2e4 = for pawns
    #
    # l1 = letter1 index
    # n1 = number1 index
    # l2 = letter2 index
    # n2 = number2 index
    #
    for l1 in range(8):
        for n1 in range(8):
            destinations = (
                # horizontal straight moves (for rook and queen)
                [(t, n1) for t in range(8)]
                # vertical straight moves (for rook and queen)
                + [(l1, t) for t in range(8)]
                # Diagonal moves (for bishop and queen)
                + [(l1 + t, n1 + t) for t in range(-7, 8)]
                + [(l1 + t, n1 - t) for t in range(-7, 8)]
                # Knight moves
                + [
                    (l1 + a, n1 + b)
                    for (a, b) in [
                        (-2, -1),
                        (-1, -2),
                        (-2, 1),
                        (1, -2),
                        (2, -1),
                        (-1, 2),
                        (2, 1),
                        (1, 2),
                    ]
                ]
            )
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + "2" + l + "1" + p)
            labels_array.append(l + "7" + l + "8" + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + "2" + l_l + "1" + p)
                labels_array.append(l + "7" + l_l + "8" + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + "2" + l_r + "1" + p)
                labels_array.append(l + "7" + l_r + "8" + p)
    return labels_array


uci_labels = create_uci_labels()
flipped_uci_labels = create_flipped_uci_labels()
unflipped_index = [uci_labels.index(x) for x in flipped_uci_labels]

move_lookup = {
    chess.Move.from_uci(move): i for move, i in zip(uci_labels, range(len(uci_labels)))
}
flipped_move_lookup = {
    chess.Move.from_uci(move): i
    for move, i in zip(flipped_uci_labels, range(len(uci_labels)))
}


def policy_from_move(move: chess.Move) -> Any:
    policy = np.zeros(len(uci_labels), dtype=np.float32)

    i = move_lookup[move]
    policy[i] = 1.0
    return policy


def flip_policy(pol: Any) -> Any:
    """

    :param pol policy to flip:
    :return: the policy, flipped (for switching between black and white it seems)
    """
    return np.asarray([pol[ind] for ind in unflipped_index])
