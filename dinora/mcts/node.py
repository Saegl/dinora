import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import chess


@dataclass
class Node:
    total_value: float
    prior: float = 0.0
    value_estimate: float = 0.0
    lazyboard: chess.Board | None = None
    move: chess.Move | None = None
    parent: Optional["Node"] = None
    children: OrderedDict[chess.Move, "Node"] = field(default_factory=OrderedDict)
    terminals: OrderedDict[chess.Move, "Node"] = field(default_factory=OrderedDict)
    number_visits: int = 0
    is_expanded: bool = False
    is_terminal: bool = False
    til_end: int = -1  # Accessible only if self.is_terminal

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

    def is_loss(self) -> bool:
        return self.value_estimate == -1.0

    def is_win(self) -> bool:
        return self.value_estimate == 1.0

    def is_draw(self) -> bool:
        return self.value_estimate == 0.0

    def to_root(self) -> None:
        if not self.lazyboard:
            assert self.parent
            assert self.move
            self.lazyboard = self.parent.board.copy()
            self.lazyboard.push(self.move)
        self.parent = None
        # Hope gc will collect parent and all his children
        # (except this one)

    def to_terminal(self) -> None:
        self.is_expanded = True
        self.is_terminal = True
        if self.parent and self.move in self.parent.children:
            # Move node from `children` to `terminals`
            assert self.move
            del self.parent.children[self.move]
            self.parent.terminals[self.move] = self

    def get_pv_line(self) -> str:
        curr = self
        line = []
        while len(curr.children) > 0 or len(curr.terminals) > 0:
            curr = curr.best()
            assert curr.move
            line.append(curr.move.uci())

        return " ".join(line)

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

    def best_terminal(self) -> "Node":
        return max(self.terminals.values(), key=lambda node: node.value_estimate)

    def best_child(self, c: float) -> "Node":
        return max(self.children.values(), key=lambda node: node.Q() + c * node.U())

    def add_child(self, move: chess.Move, prior: float, fpu: float) -> None:
        self.children[move] = Node(parent=self, move=move, prior=prior, total_value=fpu)

    def best(self) -> "Node":
        if self.is_terminal:
            if len(self.terminals) == 1:  # Node was reduced by `reduction`
                for child_terminal in self.terminals.values():
                    return child_terminal
                raise Exception("Unreachable")
            if len(self.terminals) == 0 and self.board.legal_moves.count() == 0:
                raise ValueError("Cannot get best move if there is no legal moves")
            raise Exception("Logical error, terminal must contain at most one child")

        # Use secure child here?
        # https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf
        best_non_terminal = self.best_child(1.0)

        if self.terminals:
            best_terminal = self.best_terminal()
        else:
            return best_non_terminal

        if best_non_terminal.Q() > best_terminal.value_estimate:
            return best_non_terminal
        else:
            return best_terminal

    def __str__(self) -> str:
        return f"<Node {self.move} {self.number_visits}>"
