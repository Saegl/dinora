from abc import ABC, abstractmethod
from time import time
from math import cos

extra_time = 0.5


class Constraint(ABC):
    """
    Constraints controll when to stop MCTS searching
    """

    @abstractmethod
    def meet(self) -> bool:
        pass


def time_manager(moves_number: int, time_left: int, inc: int = 0) -> float:
    moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)
    remaining_time = time_left / 1000 + moves_left * inc / 1000
    move_time = remaining_time / moves_left - extra_time
    return move_time


class TimeConstraint(Constraint):
    def __init__(
        self,
        moves_number: int,
        engine_time: int,
        engine_inc: int,
    ) -> None:
        self.move_time = time_manager(moves_number, engine_time, engine_inc)
        self.starttime = time()
        self.steps = 0

    def meet(self) -> bool:
        if self.steps < 2:  # Calculate minimum 2 nodes
            self.steps += 1
            return True
        return time() - self.starttime < self.move_time

    def __str__(self) -> str:
        return f"<TimeConstraint: {self.move_time=} {self.starttime=}>"


class NodesCountConstraint(Constraint):
    def __init__(self, count: int) -> None:
        self.step = 0
        self.count = count

    def meet(self) -> bool:
        self.step += 1
        return self.count > self.step

    def __str__(self) -> str:
        return f"<NodesCountConstraint: {self.count=} {self.step=}>"
