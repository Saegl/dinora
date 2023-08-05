import pathlib


def upload_dataset(dataset_dir: pathlib.Path, wandb_label: str):
    import wandb

    wandb.init(
        project='dinora-chess'
    )

    artifact = wandb.Artifact(wandb_label, type='dataset')
    artifact.add_dir(dataset_dir)

    wandb.log_artifact(artifact)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool to upload dataset to wandb",
    )

    parser.add_argument(
        "dataset_dir",
        help="Dataset directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "wandb_label",
        help="Wandb dataset label",
        type=str,
    )

    args = parser.parse_args()
    upload_dataset(args.dataset_dir, args.wandb_label)
