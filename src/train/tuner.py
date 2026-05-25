from __future__ import annotations

import copy
import gc
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from train.trainer import Trainer


def _is_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _gc_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _ExponentialLR(LRScheduler):
    def __init__(
        self, optimizer: Any, end_lr: float, num_iter: int, last_epoch: int = -1
    ):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        r = (self.last_epoch + 1) / self.num_iter
        if self.last_epoch > 0:
            return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
        return list(self.base_lrs)


class Tuner:
    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def _device(self) -> torch.device:
        if self.trainer.accelerator == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.trainer.accelerator)

    def lr_find(
        self,
        model: Any,
        datamodule: Any,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_training: int = 100,
        early_stop_threshold: float | None = 4.0,
        beta: float = 0.98,
    ) -> float | None:
        """LR range test (Leslie Smith). Updates model.learning_rate and returns it."""
        print(f"LR finder: {num_training} steps, {min_lr:.2e} → {max_lr:.2e}")

        device = self._device()
        model.to(device)
        model_state = copy.deepcopy(model.state_dict())

        opt_config = model.configure_optimizers()
        optimizer = opt_config["optimizer"]
        for pg in optimizer.param_groups:
            pg["lr"] = min_lr
            pg["initial_lr"] = min_lr
        scheduler = _ExponentialLR(optimizer, end_lr=max_lr, num_iter=num_training)

        lrs: list[float] = []
        losses: list[float] = []
        avg_loss = 0.0
        best_loss = float("inf")
        step = 0
        stop = False

        model.train()
        while step < num_training and not stop:
            for batch in datamodule.train_dataloader():
                if step >= num_training or stop:
                    break
                lrs.append(float(optimizer.param_groups[0]["lr"]))
                batch = self.trainer._move_batch(batch, device)
                optimizer.zero_grad()
                loss = model.training_step(batch, step)
                model._logged_metrics.clear()
                if not torch.isfinite(loss):
                    losses.append(float("nan"))
                    stop = True
                    break
                loss.backward()
                optimizer.step()
                scheduler.step()
                avg_loss = beta * avg_loss + (1 - beta) * loss.item()
                smoothed = avg_loss / (1 - beta ** (step + 1))
                if smoothed < best_loss:
                    best_loss = smoothed
                losses.append(smoothed)
                if (
                    early_stop_threshold is not None
                    and step > 1
                    and smoothed > early_stop_threshold * best_loss
                ):
                    print(f"LR finder: early stop at step {step} (loss diverged)")
                    stop = True
                    break
                step += 1

        model.load_state_dict(model_state)

        suggested = _suggest_lr(lrs, losses)
        if suggested is not None:
            model.learning_rate = suggested
            print(f"LR finder: suggested lr = {suggested:.4e}")
        else:
            print("LR finder: could not determine a suggestion, keeping original lr")
        return suggested

    def scale_batch_size(
        self,
        model: Any,
        datamodule: Any,
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        max_val: int = 8192,
    ) -> int:
        """Double batch size until OOM; return last size that fit. Updates datamodule.batch_size."""
        print(f"Batch size finder: init={init_val}, max_val={max_val}")

        device = self._device()
        model.to(device)
        model_state = copy.deepcopy(model.state_dict())

        batch_size = init_val
        last_ok = init_val

        for _ in range(max_trials):
            if batch_size > max_val:
                last_ok = min(last_ok, max_val)
                break
            datamodule.batch_size = batch_size
            if _try_steps(
                model, datamodule, device, model_state, steps_per_trial, self.trainer
            ):
                last_ok = batch_size
                print(f"  batch_size={batch_size} OK, trying {batch_size * 2}")
                batch_size *= 2
            else:
                print(f"  batch_size={batch_size} OOM, settling on {last_ok}")
                break

        model.load_state_dict(model_state)
        datamodule.batch_size = last_ok
        print(f"Batch size finder: using batch_size = {last_ok}")
        return last_ok


def _try_steps(
    model: Any,
    datamodule: Any,
    device: torch.device,
    saved_state: dict[str, Any],
    steps: int,
    trainer: Any,
) -> bool:
    """Run `steps` training steps. Returns True on success, False on OOM."""
    optimizer = model.configure_optimizers()["optimizer"]
    try:
        model.train()
        for batch in islice(datamodule.train_dataloader(), steps):
            batch = trainer._move_batch(batch, device)
            optimizer.zero_grad()
            loss = model.training_step(batch, 0)
            model._logged_metrics.clear()
            loss.backward()
            optimizer.step()
        return True
    except RuntimeError as exc:
        if _is_oom_error(exc):
            return False
        raise
    finally:
        model.load_state_dict(saved_state)
        _gc_cuda()


def _suggest_lr(
    lrs: list[float],
    losses: list[float],
    skip_begin: int = 10,
    skip_end: int = 1,
) -> float | None:
    """Return the LR at the steepest negative loss gradient."""
    lr_t = torch.tensor(lrs[skip_begin : len(lrs) - skip_end])
    loss_t = torch.tensor(losses[skip_begin : len(losses) - skip_end])
    mask = torch.isfinite(loss_t)
    lr_t, loss_t = lr_t[mask], loss_t[mask]
    if len(loss_t) < 2:
        return None
    gradients = torch.gradient(loss_t, spacing=[lr_t])[0]
    return float(lr_t[int(torch.argmin(gradients).item())])
