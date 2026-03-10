import pathlib
import sys

from dinora import PROJECT_ROOT
from dinora.models.base import BaseModel, Priors, StateValue
from dinora.models.empty_model import EmptyModel
from dinora.models.handcrafted import DummyModel

DEFAULT_ALPHANET_WEIGHTS_FILENAME = "default.ckpt"


AVAILABLE_MODELS = [
    "cached_alphanet",
    "cached_onnx",
    "cached_handcrafted",
    "alphanet",
    "onnx",
    "handcrafted",
    "empty",
]

DEFAULT_MODELS = [
    "alphanet",
    "onnx",
]


def search_weights(filename: str) -> pathlib.Path:
    app_dir = pathlib.Path(sys.argv[0]).resolve().parent

    places = [
        pathlib.Path.cwd() / filename,
        pathlib.Path.cwd() / "models" / filename,
        app_dir / filename,
        app_dir / "models" / filename,
        PROJECT_ROOT / "models" / filename,
    ]

    for place in places:
        if place.exists():
            return place

    raise Exception("Cannot find model weights")


def load_default(limit_threads: bool = False) -> BaseModel:
    for model_name in DEFAULT_MODELS:
        try:
            model = model_selector(model_name, None, None, limit_threads=limit_threads)
            return model
        except ModuleNotFoundError:
            pass
    raise Exception("No available models :-(")


def guess_model_from_weights(weights_path: pathlib.Path) -> str:
    if str(weights_path).endswith(".ckpt"):
        return "alphanet"
    elif str(weights_path).endswith(".onnx"):
        return "onnx"
    else:
        raise Exception("Unknown weights file extension")


def model_selector(  # noqa: C901
    model: str | None,
    weights_path: pathlib.Path | None,
    device: str | None,
    limit_threads: bool = False,
) -> BaseModel:
    if model is None and weights_path is not None:
        return model_selector(
            guess_model_from_weights(weights_path),
            weights_path,
            device,
            limit_threads=limit_threads,
        )

    if model is None:
        return load_default(limit_threads=limit_threads)

    if model.startswith("cached_"):
        from dinora.models.cached_model import CachedModel

        model = model.removeprefix("cached_")
        return CachedModel(
            model_selector(model, weights_path, device, limit_threads=limit_threads)
        )

    elif model == "alphanet":
        import torch  # Torch import at the top makes UCI slower

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if weights_path is None:
            weights_path = search_weights(DEFAULT_ALPHANET_WEIGHTS_FILENAME)

        # TODO: Do I really need `weights_only`?
        alphanet = torch.load(weights_path, map_location=device, weights_only=False)
        alphanet = alphanet.to(device)
        alphanet.eval()
        return alphanet  # type: ignore

    elif model == "onnx":
        from dinora.models.onnxmodel import OnnxModel

        return OnnxModel(weights_path, device, limit_threads=limit_threads)

    elif model == "handcrafted":
        return DummyModel()
    elif model == "empty":
        return EmptyModel()
    else:
        raise ValueError("Unknown model name")


__all__ = ["BaseModel", "Priors", "StateValue", "model_selector"]
