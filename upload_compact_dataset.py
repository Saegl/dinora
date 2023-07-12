import pathlib
import wandb
from dinora import PROJECT_ROOT


wandb.init(
    project='dinora-chess'
)

dataset_dir = PROJECT_ROOT / 'data' / 'converted_dataset'

artifact = wandb.Artifact('ccrl-compact', type='dataset')
artifact.add_dir(dataset_dir)

wandb.log_artifact(artifact)
