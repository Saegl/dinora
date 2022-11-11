import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import chess


@dataclass
class Node:
    total_value: float
    prior: float = 0.0
    board_value_estimate_info: float = (
        -99.0
    )  # NOT used in search calc, only for treeviz info
    lazyboard: chess.Board | None = None
    move: chess.Move | None = None
    parent: Optional["Node"] = None
    children: OrderedDict[chess.Move, "Node"] = field(default_factory=OrderedDict)
    number_visits: int = 0
    is_expanded: bool = False

    @property
    def board(self) -> chess.Board:
        """Lazyboard loading, copy parent board only if getter called"""
        if not self.lazyboard:
            # Root always has lazyboard
            assert self.parent
            assert self.move
            self.lazyboard = self.parent.board.copy()
            self.lazyboard.push(self.move)
        return self.lazyboard

    def is_root(self) -> bool:
        return self.parent is None

    def get_most_visited_node(self) -> "Node":
        _, node = max(
            self.children.items(),
            key=lambda item: (item[1].number_visits, item[1].Q()),
        )
        return node

    def get_highest_q_node(self) -> "Node":
        _, node = max(
            self.children.items(),
            key=lambda item: (item[1].Q(), item[1].number_visits),
        )
        return node

    def Q(self) -> float:
        return self.total_value / (1 + self.number_visits)

    def U(self) -> float:
        assert self.parent
        return (
            math.sqrt(self.parent.number_visits) * self.prior / (1 + self.number_visits)
        )

    def best_child(self, c: float) -> "Node":
        return max(self.children.values(), key=lambda node: node.Q() + c * node.U())

    def add_child(self, move: chess.Move, prior: float, fpu: float) -> None:
        self.children[move] = Node(parent=self, move=move, prior=prior, total_value=fpu)

    def __str__(self) -> str:
        return f"UCTNode <{self.board}, {self.children}>"
