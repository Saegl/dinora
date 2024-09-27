import pathlib


def upload_dataset(dataset_dir: pathlib.Path, wandb_label: str) -> None:
    import wandb

    wandb.init(project="dinora-chess")

    artifact = wandb.Artifact(wandb_label, type="dataset")
    artifact.add_dir(dataset_dir)  # type: ignore

    wandb.log_artifact(artifact)
