from dinora.models.base import BaseModel
from dinora.models.registry import ModelConfig, search_weights

DEFAULT_WEIGHTS_FILENAME = "default.ckpt"


def load(config: ModelConfig) -> BaseModel:
    """
    Load a checkpoint saved by `src/train`.

    The name says `torch`, not an architecture: `src/train` saves whole modules
    (`torch.save(model)`), so unpickling rebuilds the class and whichever network
    was trained (`AlphaNet`, `SeNet`, ...) comes back. A state dict would not be
    enough, nothing here says which architecture and hyperparameters to build
    before filling in the weights.
    """
    import torch  # Torch import at the top makes UCI slower

    device = config.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = config.weights
    if weights is None:
        weights = search_weights(DEFAULT_WEIGHTS_FILENAME)

    # Rebuilding the pickled class is exactly what `weights_only=True` forbids
    # ("Unsupported global: GLOBAL dinora.models.senet.SeNet"), and allowlisting
    # it does not help: the next blocker is `builtins.set`, then the rest of the
    # module internals. So this stays `False`, which means unpickling runs
    # arbitrary code from the file - only load weights you trust. Torch 2.6 made
    # `True` the default, so passing it explicitly is what survives an upgrade.
    model = torch.load(weights, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    assert isinstance(model, BaseModel)
    return model
