from operator import attrgetter
from dinora.mcts import Node


def reduce_parent(parent: Node, child: Node):
    parent.to_terminal()
    parent.til_end = child.til_end + 1
    parent.value_estimate = -child.value_estimate
    parent.children.clear()
    parent.terminals.clear()
    parent.terminals[child.move] = child


def reduce_to_win(node: Node):
    at_least_one_lost = any(map(Node.is_loss, node.terminals.values()))
    if at_least_one_lost:
        return min(
            filter(Node.is_loss, node.terminals.values()), key=attrgetter("til_end")
        )


def reduce_to_loss(node: Node):
    only_terminals_left = len(node.children) == 0
    if not only_terminals_left:
        return None

    all_won = all(map(Node.is_win, node.terminals.values()))
    if all_won:
        return max(
            filter(Node.is_win, node.terminals.values()), key=attrgetter("til_end")
        )


def reduce_to_draw(node: Node):
    only_terminals_left = len(node.children) == 0
    all_draws = all(map(Node.is_draw, node.terminals.values()))
    at_least_one_win = any(map(Node.is_win, node.terminals.values()))
    at_least_one_draw = any(map(Node.is_draw, node.terminals.values()))

    if only_terminals_left and all_draws or at_least_one_win and at_least_one_draw:
        return max(
            filter(Node.is_draw, node.terminals.values()), key=attrgetter("til_end")
        )


def reduction(node: Node):
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
