import chess

from dinora.mcts.node import Node
from dinora.mcts.uci_info import UciInfo
from dinora.mcts.noise import apply_noise
from dinora.mcts.constraints import Constraint
from dinora.mcts.params import MCTSparams
from dinora.models import BaseModel, Priors


def selection(root: Node, c: float) -> Node:
    current = root

    while current.is_expanded and current.children:
        current = current.best_child(c)

    return current


def reduction(node: Node):
    current = node
    while current.is_terminal and current.parent:
        current = current.parent
        if current.is_terminal:
            continue

        # Current is not terminal, Can we make it terminal?
        #### Try to prove win: there is at least one child lost

        loss_child = None

        for child in current.terminals.values():
            if (
                child.value_estimate == -1.0
                and
                # Find fastest way to win
                (loss_child is None or child.til_end < loss_child.til_end)
            ):
                loss_child = child

        if loss_child:
            current.to_terminal()
            current.til_end = loss_child.til_end + 1
            current.value_estimate = 1.0
            current.children.clear()
            current.terminals.clear()
            current.terminals[loss_child.move] = loss_child
            continue

        #### Try to prove loss: all chilren are won

        if len(current.children) > 0:
            continue  # cannot prove loss if there is non terminals

        if len(current.terminals) == 0:
            raise Exception(
                "logical error: at this state, current must be terminal "
                "and have at least one terminal child"
            )

        all_won = True
        win_child = None
        for child in current.terminals.values():
            if child.value_estimate == 1.0:
                # Find slowest way to lost
                if win_child is None or child.til_end > win_child.til_end:
                    win_child = child
            else:
                all_won = False
                break

        if all_won:
            current.to_terminal()
            current.til_end = win_child.til_end + 1
            current.value_estimate = -1.0
            current.children.clear()
            current.terminals.clear()
            current.terminals[win_child.move] = win_child
            continue

    return node


def expansion(node: Node, child_priors: Priors, fpu: float) -> None:
    node.is_expanded = True

    for move, prior in child_priors.items():
        node.add_child(move, prior, fpu)


def backpropagation(node: Node, value_estimate: float) -> None:
    current = node
    # Child nodes are multiplied by -1 because we want max(-opponent eval)
    turnfactor = -1.0
    while current.parent is not None:
        current.number_visits += 1
        current.total_value += value_estimate * turnfactor
        current = current.parent
        turnfactor *= -1
    current.number_visits += 1


def run_mcts(
    state: chess.Board | Node,
    constraint: Constraint,
    evaluator: BaseModel,
    params: MCTSparams,
) -> Node:
    uci_info = UciInfo()

    # Can reuse tree
    if isinstance(state, Node):
        root = state
        uci_info.reuse_stats(root.number_visits, params.send_func)
        print(root)
    else:
        root = Node(params.fpu_at_root, lazyboard=state)
        root.is_terminal, child_priors, root.value_estimate = evaluator.evaluate(
            root.board
        )
        child_priors = apply_noise(
            child_priors,
            params.dirichlet_alpha,
            params.noise_eps,
        )
        expansion(root, child_priors, params.fpu)
        backpropagation(root, root.value_estimate)

    while constraint.meet():
        leaf = selection(root, params.cpuct)

        is_terminal, child_priors, leaf.value_estimate = evaluator.evaluate(leaf.board)

        if is_terminal:
            leaf.to_terminal()
            leaf.til_end = 0
            leaf = reduction(leaf)

        if not leaf.is_terminal and not leaf.is_expanded:
            expansion(leaf, child_priors, params.fpu)

        backpropagation(leaf, leaf.value_estimate)
        uci_info.after_iteration(root, params.send_func)

        if root.is_terminal:
            break

    uci_info.at_mcts_end(root, params.send_func)
    return root
