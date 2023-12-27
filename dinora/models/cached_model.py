import chess
import pylru
from dinora.models.base import BaseModel, IsTerminal, Priors, StateValue


class CachedModel(BaseModel):
    def __init__(self, model: BaseModel, size: int = 200000) -> None:
        self.cache = pylru.lrucache(size)
        self.model = model
        self.hit = 0
        self.misses = 0

    def clear(self) -> None:
        self.hit = 0
        self.misses = 0
        self.cache.clear()

    @property
    def hit_ratio(self) -> float:
        return self.hit / (self.hit + self.misses)

    def evaluate(self, board: chess.Board) -> tuple[IsTerminal, Priors, StateValue]:
        epd = board.epd()
        if epd in self.cache:
            self.hit += 1
            is_terminal, policy, value = self.cache[epd]
            return is_terminal, policy, value
        else:
            self.misses += 1
            is_terminal, policy, value = self.model.evaluate(board)
            self.cache[epd] = [is_terminal, policy, value]
            return is_terminal, policy, value

    def reset(self):
        self.clear()
        self.model.reset()
