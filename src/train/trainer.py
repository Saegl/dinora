from __future__ import annotations

import pathlib
import time
from datetime import timedelta
from itertools import islice
from typing import Any

import torch

from train.callback import Callback


class Trainer:
    def __init__(
        self,
        max_epochs: int = 1,
        max_time: timedelta | None = None,
        logger: Any = None,
        log_every_n_steps: int = 50,
        enable_checkpointing: bool = False,
        default_root_dir: pathlib.Path | None = None,
        callbacks: list[Callback] | None = None,
        val_check_interval: float | int = 1.0,
        limit_train_batches: int | None = None,
        limit_val_batches: int | None = None,
        limit_test_batches: int | None = None,
        num_sanity_val_steps: int = 2,
        accelerator: str = "auto",
    ) -> None:
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self.enable_checkpointing = enable_checkpointing
        self.default_root_dir = default_root_dir
        self.callbacks = callbacks or []
        self.val_check_interval = val_check_interval
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.num_sanity_val_steps = num_sanity_val_steps
        self.accelerator = accelerator

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.optimizers: list[Any] = []

        self._model: Any = None
        self._optimizer: Any = None
        self._scheduler: Any = None

        # 0 means no validation at all, so the callbacks don't fire either
        self._validation_enabled = limit_val_batches != 0

    def _move_batch(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, list | tuple):
            moved = [self._move_batch(b, device) for b in batch]
            return type(batch)(moved)
        return batch

    def _flush_metrics(self, model: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = getattr(model, "_logged_metrics", {}).copy()
        if hasattr(model, "_logged_metrics"):
            model._logged_metrics.clear()
        return metrics

    def _log_train_metrics(self, model: Any, force: bool = False) -> None:
        should_log = force or self.global_step % self.log_every_n_steps == 0
        if should_log and self.logger:
            metrics = self._flush_metrics(model)
            if metrics:
                metrics["epoch"] = self.current_epoch
                self.logger.log_metrics(metrics, step=self.global_step)
        else:
            self._flush_metrics(model)

    def _run_validation(self, model: Any, val_loader: Any) -> None:
        if not self._validation_enabled:
            return
        model.eval()
        device = next(model.parameters()).device
        all_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            loader: Any = enumerate(val_loader)
            if self.limit_val_batches is not None:
                loader = islice(loader, self.limit_val_batches)
            for batch_idx, batch in loader:
                batch = self._move_batch(batch, device)
                model.validation_step(batch, batch_idx)
                for k, v in self._flush_metrics(model).items():
                    all_metrics.setdefault(k, []).append(float(v))
        if all_metrics and self.logger:
            avg = {k: sum(vs) / len(vs) for k, vs in all_metrics.items()}
            avg["epoch"] = self.current_epoch
            self.logger.log_metrics(avg, step=self.global_step)

    def _run_sanity_check(self, model: Any, val_loader: Any) -> None:
        # Fires on_validation_end callbacks without logging validation/* metrics,
        # which is what gives callback series like cploss/* their point at step 0.
        if self.num_sanity_val_steps <= 0 or not self._validation_enabled:
            return
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            loader: Any = islice(enumerate(val_loader), self.num_sanity_val_steps)
            for batch_idx, batch in loader:
                batch = self._move_batch(batch, device)
                model.validation_step(batch, batch_idx)
                self._flush_metrics(model)  # discard, like sanity_checking
        for callback in self.callbacks:
            callback.on_validation_end(self, model)
        model.train()

    def _run_train_epoch(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        scheduler_frequency: int,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        start_time: float,
    ) -> bool:
        loader: Any = enumerate(train_loader)
        if self.limit_train_batches is not None:
            loader = islice(loader, self.limit_train_batches)

        batches_in_epoch = (
            min(len(train_loader), self.limit_train_batches)
            if self.limit_train_batches is not None
            else len(train_loader)
        )

        if isinstance(self.val_check_interval, float):
            val_every = max(1, int(batches_in_epoch * self.val_check_interval))
        else:
            val_every = int(self.val_check_interval)

        for steps_this_epoch, (batch_idx, batch) in enumerate(loader, start=1):
            for callback in self.callbacks:
                callback.on_train_batch_start(self, model, batch, batch_idx)

            batch = self._move_batch(batch, device)
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            self.global_step += 1

            if steps_this_epoch % scheduler_frequency == 0:
                scheduler.step()

            stopping = (
                self.max_time is not None
                and time.time() - start_time >= self.max_time.total_seconds()
            )

            self._log_train_metrics(model, force=stopping)

            if self._validation_enabled and (
                stopping or steps_this_epoch % val_every == 0
            ):
                self._run_validation(model, val_loader)
                for callback in self.callbacks:
                    callback.on_validation_end(self, model)
                model.train()

            if stopping:
                return True

        return False

    def _restore_checkpoint(
        self, path: pathlib.Path, model: Any, optimizer: Any, scheduler: Any
    ) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Fall back to pre-migration key names so old checkpoints still load.
        model_state = state.get("model_state_dict")
        if model_state is None:
            model_state = state["state_dict"]
        model.load_state_dict(model_state)

        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is None and state.get("optimizer_states"):
            optimizer_state = state["optimizer_states"][0]
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        scheduler_state = state.get("scheduler_state_dict")
        if scheduler_state is None and state.get("lr_schedulers"):
            scheduler_state = state["lr_schedulers"][0]
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        self.global_step = state.get("global_step", 0)
        self.current_epoch = state.get("epoch", 0)
        print(f"Restored {path}: epoch {self.current_epoch}, step {self.global_step}")

    def fit(
        self,
        model: Any,
        datamodule: Any,
        ckpt_path: pathlib.Path | None = None,
    ) -> None:
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config["optimizer"]
        self.optimizers = [optimizer]
        self._model = model
        self._optimizer = optimizer

        scheduler_cfg = optimizer_config["lr_scheduler"]
        scheduler = scheduler_cfg["scheduler"]
        scheduler_frequency: int = scheduler_cfg["frequency"]
        self._scheduler = scheduler

        if self.accelerator == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.accelerator)
        model.to(device)
        print(f"Training on {device}")

        # after `model.to`, so the optimizer state lands on the training device
        if ckpt_path is not None:
            self._restore_checkpoint(ckpt_path, model, optimizer, scheduler)

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        for callback in self.callbacks:
            callback.on_fit_start(self, model)

        self._run_sanity_check(model, val_loader)

        start_time = time.time()
        max_epochs = self.max_epochs if self.max_epochs >= 0 else 10**9

        done = False
        for epoch in range(self.current_epoch, max_epochs):
            if done:
                break
            if (
                self.max_time is not None
                and time.time() - start_time >= self.max_time.total_seconds()
            ):
                break

            self.current_epoch = epoch
            model.train()
            done = self._run_train_epoch(
                model,
                optimizer,
                scheduler,
                scheduler_frequency,
                train_loader,
                val_loader,
                device,
                start_time,
            )

        for callback in self.callbacks:
            callback.on_train_end(self, model)

    def validate(self, model: Any, datamodule: Any) -> None:
        val_loader = datamodule.val_dataloader()
        self._run_validation(model, val_loader)

    def save_checkpoint(self, path: pathlib.Path) -> None:
        state: dict[str, Any] = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
        }
        if self._model is not None:
            state["model_state_dict"] = self._model.state_dict()
        if self._optimizer is not None:
            state["optimizer_state_dict"] = self._optimizer.state_dict()
        if self._scheduler is not None:
            state["scheduler_state_dict"] = self._scheduler.state_dict()
        torch.save(state, path)
