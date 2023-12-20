import pathlib


def upload_dataset(dataset_dir: pathlib.Path, wandb_label: str):
    import wandb

    wandb.init(project="dinora-chess")

    artifact = wandb.Artifact(wandb_label, type="dataset")
    artifact.add_dir(dataset_dir)

    wandb.log_artifact(artifact)
