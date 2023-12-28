import pathlib

from dinora import DEFAULT_WEIGHTS
from dinora.models.base import BaseModel, IsTerminal, Priors, StateValue
from dinora.models.cached_model import CachedModel
from dinora.models.handcrafted import DummyModel


def model_selector(
    model: str, weights_path: pathlib.Path | None, device: str | None
) -> BaseModel:
    if model.startswith("cached_"):
        model = model.removeprefix("cached_")
        return CachedModel(model_selector(model, weights_path, device))

    elif model == "alphanet":
        import torch  # Torch import at the top makes UCI slower

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS

        alphanet = torch.load(weights_path, map_location=device)
        alphanet = alphanet.to(device)
        return alphanet

    elif model == "onnx":
        from dinora.models.onnxmodel import OnnxModel

        return OnnxModel(weights_path, device)

    elif model == "handcrafted":
        return DummyModel()
    else:
        raise ValueError("Unknown model name")


__all__ = ["BaseModel", "IsTerminal", "Priors", "StateValue", "model_selector"]
