import threading
from math import cos
from time import time

MS_TO_S = 1 / 1000


class Stopper:
    """
    Stoppers control when to stop searching
    """

    def __init__(self) -> None:
        self.early_stop = threading.Event()
        self.called = False

    def should_stop(self) -> bool:
        if not self.called:  # Calculate minimum 1 node
            self.called = True
            return False
        return self.early_stop.is_set()


class Time(Stopper):
    def __init__(
        self,
        moves_number: int,
        engine_time: int,
        engine_inc: int,
        move_overhead: int,
    ) -> None:
        super().__init__()
        self.movetime = (
            self.calc_movetime(moves_number, engine_time, engine_inc, move_overhead)
            * MS_TO_S
        )
        self.starttime = time()

    @staticmethod
    def calc_movetime(
        moves_number: int, time_left: int, inc: int, move_overhead: int
    ) -> float:
        # TODO: this one made in desmos, should be simpler
        moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)

        remaining_time = time_left + moves_left * inc
        movetime = remaining_time / moves_left - move_overhead
        return movetime

    def should_stop(self) -> bool:
        if not self.called:
            self.called = True
            return False
        if super().should_stop():
            return True
        return time() - self.starttime > self.movetime

    def __str__(self) -> str:
        return f"<Time: {self.movetime=} {self.starttime=}>"


class MoveTime(Stopper):
    movetime: float
    starttime: float

    def __init__(self, movetime: int) -> None:
        super().__init__()
        self.movetime = movetime * MS_TO_S
        self.starttime = time()

    def should_stop(self) -> bool:
        if not self.called:
            self.called = True
            return False
        if super().should_stop():
            return True
        return time() - self.starttime > self.movetime

    def __str__(self) -> str:
        return f"<MoveTime: {self.movetime=} {self.starttime=}>"


class NodesCount(Stopper):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.step = 0
        self.count = count

    def should_stop(self) -> bool:
        if super().should_stop():
            return True
        self.step += 1
        return self.count < self.step

    def __str__(self) -> str:
        return f"<NodesCount: {self.count=} {self.step=}>"


class Infinite(Stopper):
    def __init__(self) -> None:
        super().__init__()

    def should_stop(self) -> bool:
        return super().should_stop()

    def __str__(self) -> str:
        return "<Infinite 8>"
