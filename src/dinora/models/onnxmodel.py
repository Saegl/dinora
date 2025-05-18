import pathlib

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime

from dinora.encoders.board_tensor import boards_to_tensor
from dinora.encoders.policy import legal_policy
from dinora.models import search_weights
from dinora.models.base import BaseModel, Evaluation

npf32 = npt.NDArray[np.float32]
DEFAULT_WEIGHTS_FILENAME = "alphanet_classic.ckpt.onnx"


class OnnxModel(BaseModel):
    def __init__(self, weights: pathlib.Path | None = None, device: str | None = None):
        if device is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device '{device}' is not supported")

        if weights is None:
            weights = search_weights(DEFAULT_WEIGHTS_FILENAME)

        self.ort_session = onnxruntime.InferenceSession(weights, providers=providers)
        self.weights_path = weights

    def name(self) -> str:
        classname = self.__class__.__name__
        providers = self.ort_session.get_providers()
        return f"{classname} {providers} {self.weights_path}"

    def inference_np(self, batch_np: npf32) -> tuple[npf32, npf32]:
        raw_policy, raw_value = self.ort_session.run(None, {"input": batch_np})
        return raw_policy, raw_value

    def evaluate(self, board: chess.Board) -> Evaluation:
        return self.evaluate_batch([board])[0]

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        raw_policy, raw_value = self.inference_np(boards_to_tensor(boards))

        return [
            (legal_policy(raw_policy[i], board), float(raw_value[i, 0]))
            for i, board in enumerate(boards)
        ]
