"""
Model loading: the registry, the loader contract and weights lookup.

`ModelSpec.load` is a `"module:attribute"` import spec, not a callable, so that
importing this module stays free of numpy & co. See `dinora.threads`.
"""

import importlib
import importlib.util
import pathlib
import sys
import warnings
from dataclasses import dataclass

from dinora import PROJECT_ROOT
from dinora.models.base import BaseModel


@dataclass(frozen=True)
class ModelConfig:
    """
    Everything a model loader may need.

    Every loader in the registry takes this and nothing else, so the registry
    can call any of them without knowing which knobs they care about.
    """

    name: str | None = None
    weights: pathlib.Path | None = None
    device: str | None = None
    limit_threads: bool = False


@dataclass(frozen=True)
class ModelSpec:
    load: str
    """`"module:attribute"` of a `(ModelConfig) -> BaseModel` loader"""

    requires: str | None = None
    """Third party package to check for before falling back to this model"""


MODELS: dict[str, ModelSpec] = {
    "torch": ModelSpec("dinora.models.torchmodel:load", requires="torch"),
    "onnx": ModelSpec("dinora.models.onnxmodel:load", requires="onnxruntime"),
    "handcrafted": ModelSpec("dinora.models.handcrafted:load"),
    "empty": ModelSpec("dinora.models.empty_model:load"),
}

# Tried in order when neither a model name nor weights are given
FALLBACK_ORDER = ["torch", "onnx"]

WEIGHTS_SUFFIX_TO_MODEL = {".ckpt": "torch", ".onnx": "onnx"}


class UnknownModel(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Unknown model '{name}', available: {', '.join(sorted(MODELS))}"
        )


class UnknownWeightsFormat(Exception):
    def __init__(self, weights: pathlib.Path) -> None:
        super().__init__(
            f"Cannot tell which model loads '{weights}',"
            f" known extensions: {', '.join(sorted(WEIGHTS_SUFFIX_TO_MODEL))}"
        )


class WeightsNotFound(Exception):
    def __init__(self, filename: str, places: list[pathlib.Path]) -> None:
        looked_in = "\n".join(f"  {place}" for place in places)
        super().__init__(f"Cannot find weights '{filename}', looked in:\n{looked_in}")


class NoModelAvailable(Exception):
    def __init__(self, failures: list[str]) -> None:
        reasons = "\n".join(f"  {failure}" for failure in failures)
        super().__init__(f"No usable model found, tried:\n{reasons}")


def search_places(filename: str) -> list[pathlib.Path]:
    app_dir = pathlib.Path(sys.argv[0]).resolve().parent

    return [
        pathlib.Path.cwd() / filename,
        pathlib.Path.cwd() / "models" / filename,
        app_dir / filename,
        app_dir / "models" / filename,
        PROJECT_ROOT / "models" / filename,
    ]


def search_weights(filename: str) -> pathlib.Path:
    places = search_places(filename)

    for place in places:
        if place.exists():
            return place

    raise WeightsNotFound(filename, places)


def model_from_weights(weights: pathlib.Path) -> str:
    try:
        return WEIGHTS_SUFFIX_TO_MODEL[weights.suffix]
    except KeyError:
        raise UnknownWeightsFormat(weights) from None


def _load(name: str, config: ModelConfig) -> BaseModel:
    if name not in MODELS:
        raise UnknownModel(name)

    module_name, _, attribute = MODELS[name].load.partition(":")
    load = getattr(importlib.import_module(module_name), attribute)
    model: BaseModel = load(config)
    return model


def _load_fallback(config: ModelConfig) -> BaseModel:
    """
    Load the first model of `FALLBACK_ORDER` that works.

    A backend can be missing (extra not installed) or installed but unusable
    (stale checkpoint, no weights). Neither is silently ignored: whatever went
    wrong ends up either in a warning or in the final `NoModelAvailable`.
    """
    failures: list[str] = []

    for name in FALLBACK_ORDER:
        requires = MODELS[name].requires
        if requires is not None and importlib.util.find_spec(requires) is None:
            failures.append(f"{name}: '{requires}' is not installed")
            continue

        try:
            model = _load(name, config)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        if failures:
            warnings.warn(
                f"Fell back to model '{name}', earlier candidates failed: "
                + "; ".join(failures),
                stacklevel=2,
            )
        return model

    raise NoModelAvailable(failures)


def build_model(config: ModelConfig) -> BaseModel:
    name = config.name

    if name is None and config.weights is not None:
        name = model_from_weights(config.weights)

    if name is None:
        return _load_fallback(config)

    return _load(name, config)
