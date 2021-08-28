import chess
import numpy as np


def create_flipped_uci_labels():
    """
    Seems to somehow transform the labels used for describing the universal chess interface format, putting
    them into a returned list.
    :return:
    """

    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in create_uci_labels()]


def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8"]
    promoted_to = ["q", "r", "b", "n"]

    for l1 in range(8):
        for n1 in range(8):
            destinations = (
                [(t, n1) for t in range(8)]
                + [(l1, t) for t in range(8)]
                + [(l1 + t, n1 + t) for t in range(-7, 8)]
                + [(l1 + t, n1 - t) for t in range(-7, 8)]
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


def policy_from_move(move: chess.Move):
    policy = np.zeros(len(uci_labels))

    i = move_lookup[move]
    policy[i] = 1.0
    return policy


def flip_policy(pol):
    """

    :param pol policy to flip:
    :return: the policy, flipped (for switching between black and white it seems)
    """
    return np.asarray([pol[ind] for ind in unflipped_index])


if __name__ == "__main__":
    move = chess.Move.from_uci("e2e4")
    print(uci_labels)
    print(flipped_uci_labels)

    print(move_lookup[move])
    print(flipped_move_lookup[move])
