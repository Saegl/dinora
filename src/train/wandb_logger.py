import pathlib
from typing import Any

import wandb


class WandbLogger:
    """Logs metrics and checkpoints to wandb.

    ``trainer/global_step`` is logged as a value and declared as the x-axis for every
    metric, so charts plot against the training step rather than wandb's ``_step``;
    metrics logged without a step (cploss, elofish) attach to the latest one via
    ``step_sync``.
    """

    def __init__(
        self,
        project: str,
        log_model: str | bool | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project = project
        self.log_model = log_model
        self._checkpoint_name: str | None = None
        self._metrics_defined = False
        self._define_metrics()
        if config is not None and wandb.run is not None:
            wandb.run.config.update(config)

    def _define_metrics(self) -> None:
        if self._metrics_defined or wandb.run is None:
            return
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step", step_sync=True)
        self._metrics_defined = True

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._define_metrics()
        if step is not None:
            wandb.log({**metrics, "trainer/global_step": step})
        else:
            wandb.log(metrics)

    def log_text(self, key: str, columns: list[str], data: list[Any]) -> None:
        self._define_metrics()
        wandb.log({key: wandb.Table(columns=columns, data=data)})  # type: ignore[no-untyped-call]

    def after_save_checkpoint(
        self, filepath: pathlib.Path, metadata: dict[str, Any] | None = None
    ) -> None:
        """Upload a saved checkpoint when ``log_model`` is enabled."""
        if self.log_model not in ("all", True) or wandb.run is None:
            return
        if self._checkpoint_name is None:
            self._checkpoint_name = f"model-{wandb.run.id}"
        artifact = wandb.Artifact(
            name=self._checkpoint_name, type="model", metadata=metadata
        )
        artifact.add_file(str(filepath), name="model.ckpt")
        wandb.run.log_artifact(artifact, aliases=["latest", "best"])
