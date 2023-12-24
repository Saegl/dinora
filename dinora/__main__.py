import warnings
from dinora.cli import run_cli

# torchvision is incorrectly installed by conda, but it is not needed anyway
warnings.filterwarnings("ignore", message="Failed to load image Python extension:*")

run_cli()
