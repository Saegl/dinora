import pathlib

import torch
import torch.onnx

from dinora.models.registry import ModelConfig, build_model


def export_onnx(model_name: str, weights: pathlib.Path) -> None:
    model = build_model(ModelConfig(model_name, weights, "cpu"))
    assert isinstance(model, torch.nn.Module), "Can convert only `torch` models"

    x = torch.randn(1, 18, 8, 8, requires_grad=False)

    torch.onnx.export(
        model,
        x,
        str(weights.parent / (weights.name + ".onnx")),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )
