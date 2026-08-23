import pathlib

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from dinora.encoders.board_tensor import boards_to_tensor
from dinora.encoders.policy import legal_policy
from dinora.models.base import BaseModel, Evaluation
from dinora.models.registry import ModelConfig, search_weights

npf32 = npt.NDArray[np.float32]
DEFAULT_WEIGHTS_FILENAME = "default.onnx"


class OnnxModel(BaseModel):
    def __init__(
        self,
        weights: pathlib.Path | None = None,
        device: str | None = None,
        limit_threads: bool = False,
    ):
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

        sess_options = ort.SessionOptions()
        if limit_threads:
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1

        self.ort_session = ort.InferenceSession(
            weights, providers=providers, sess_options=sess_options
        )
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


def load(config: ModelConfig) -> BaseModel:
    return OnnxModel(config.weights, config.device, limit_threads=config.limit_threads)
