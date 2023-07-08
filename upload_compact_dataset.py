import pathlib
import wandb


PROJECT_DIR = pathlib.Path(__file__).parent


wandb.init(
    project='dinora-chess'
)

dataset_dir = PROJECT_DIR / 'data' / 'converted_dataset'

artifact = wandb.Artifact('ccrl-compact', type='dataset')
artifact.add_dir(dataset_dir)

wandb.log_artifact(artifact)
