from time import time
from typing import Callable

from dinora.mcts.node import Node


def cp(q: float) -> int:
    """Convert UCT Q values to Stockfish like centipawns"""
    return int(295 * q / (1 - 0.976953125 * q**14))


def calc_score(node: Node) -> int:
    score = int(round(cp(node.Q()), 0))
    return score


def send_info(
    send: Callable[[str], None],
    bestmove: str,
    count: int,
    delta: float,
    score: int,
) -> None:
    if send != None:
        send(
            "info depth 1 seldepth 1 score cp {} nodes {} nps {} pv {}".format(
                score, count, int(round(count / delta, 0)), bestmove
            )
        )


def send_tree_info(send: Callable[[str], None], root: Node) -> None:
    if send != None:
        for nd in sorted(root.children.items(), key=lambda item: item[1].number_visits):
            send(
                "info string {} {} \t(P: {}%) \t(Q: {})".format(
                    nd[1].move,
                    nd[1].number_visits,
                    round(nd[1].prior * 100, 2),
                    round(nd[1].Q(), 5),
                )
            )


class UciInfo:
    def __init__(self) -> None:
        self.start_time = time()
        self.count = 0
        self.delta_last = 0.0

    def after_iteration(self, root: Node, send_func: Callable[[str], None]) -> None:
        self.count += 1
        now = time()
        delta = now - self.start_time

        if delta - self.delta_last > 5:  # Send info every 5 sec
            self.delta_last = delta
            bestnode = root.get_most_visited_node()
            score = calc_score(bestnode)

            assert bestnode.move
            send_info(
                send_func,
                bestnode.move.uci(),
                self.count,
                delta,
                score,
            )

    def at_mcts_end(self, root: Node, send_func: Callable[[str], None]) -> None:
        bestnode = root.get_most_visited_node()
        delta = time() - self.start_time
        score = calc_score(bestnode)

        assert bestnode.move
        send_info(send_func, bestnode.move.uci(), self.count, delta, score)
        send_tree_info(send_func, root)
