import pathlib

from dinora.models.base import BaseModel, Priors, StateValue
from dinora.models.cached_model import CachedModel
from dinora.models.handcrafted import DummyModel


def model_selector(model: str, weights_path: pathlib.Path) -> BaseModel:
    if model.startswith("cached_"):
        model = model.removeprefix("cached_")
        return CachedModel(model_selector(model))

    elif model == "alphanet":
        import torch  # Torch import at the top makes UCI slower

        return torch.load(weights_path)

    elif model == "handcrafted":
        return DummyModel()
    else:
        raise ValueError("Unknown model name")


__all__ = ["BaseModel", "Priors", "StateValue", "model_selector"]
