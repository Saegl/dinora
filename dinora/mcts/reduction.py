from operator import attrgetter

import chess
from dinora.mcts import Node


def terminal_val(board: chess.Board) -> float | None:
    outcome = board.outcome()

    if outcome is not None and outcome.winner is not None:
        # There is a winner, but is is our turn to move
        # means we lost
        return -1.0

    elif (
        outcome is not None  # We know there is no winner by `if` above, so it's draw
        or board.is_repetition(3)  # Can claim draw means it's draw
        or board.can_claim_fifty_moves()
    ):
        # There is some subtle difference between
        # board.can_claim_threefold_repetition() and board.is_repetition(3)
        # I believe it is right to use second one, but I am not 100% sure
        # Gives +-1 ply error on this one for example https://lichess.org/EJ67iHS1/black#94

        return 0.0
    else:
        assert outcome is None
        return None


def reduce_parent(parent: Node, child: Node) -> None:
    parent.to_terminal()
    parent.til_end = child.til_end + 1
    parent.value_estimate = -child.value_estimate
    parent.children.clear()
    parent.terminals.clear()
    assert child.move
    parent.terminals[child.move] = child


def reduce_to_win(node: Node) -> Node | None:
    at_least_one_lost = any(map(Node.is_loss, node.terminals.values()))
    if at_least_one_lost:
        return min(
            filter(Node.is_loss, node.terminals.values()), key=attrgetter("til_end")
        )
    return None


def reduce_to_loss(node: Node) -> Node | None:
    only_terminals_left = len(node.children) == 0
    if not only_terminals_left:
        return None

    all_won = all(map(Node.is_win, node.terminals.values()))
    if all_won:
        return max(
            filter(Node.is_win, node.terminals.values()), key=attrgetter("til_end")
        )

    return None


def reduce_to_draw(node: Node) -> Node | None:
    only_terminals_left = len(node.children) == 0
    all_draws = all(map(Node.is_draw, node.terminals.values()))
    at_least_one_win = any(map(Node.is_win, node.terminals.values()))
    at_least_one_draw = any(map(Node.is_draw, node.terminals.values()))

    if only_terminals_left and all_draws or at_least_one_win and at_least_one_draw:
        return max(
            filter(Node.is_draw, node.terminals.values()), key=attrgetter("til_end")
        )
    return None


def reduction(node: Node) -> Node:
    current = node
    while current.is_terminal and current.parent:
        current = current.parent
        if current.is_terminal:
            continue

        if child := reduce_to_win(current):
            reduce_parent(current, child)
            continue

        if child := reduce_to_loss(current):
            reduce_parent(current, child)
            continue

        if child := reduce_to_draw(current):
            reduce_parent(current, child)
            continue

    return node
