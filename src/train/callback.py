from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from train.trainer import Trainer


class Callback:
    def on_fit_start(self, trainer: Trainer, model: Any) -> None:
        pass

    def on_train_batch_start(
        self, trainer: Trainer, model: Any, batch: Any, batch_idx: int
    ) -> None:
        pass

    def on_validation_end(self, trainer: Trainer, model: Any) -> None:
        pass

    def on_train_end(self, trainer: Trainer, model: Any) -> None:
        pass
