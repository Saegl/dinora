import pathlib
import numpy as np
import numpy.typing as npt

import chess
import onnxruntime

from dinora import PROJECT_ROOT
from dinora.models.nnwrapper import NNWrapper
from dinora.encoders.board_representation import board_to_tensor


npf32 = npt.NDArray[np.float32]
DEFAULT_ONNX_WEIGHTS = PROJECT_ROOT / "models/alphanet_classic.ckpt.onnx"


class OnnxModel(NNWrapper):
    def __init__(self, weights: pathlib.Path | None, device: str | None):
        if device is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device '{device}' is not supported")

        if weights is None:
            weights = DEFAULT_ONNX_WEIGHTS

        self.ort_session = onnxruntime.InferenceSession(weights, providers=providers)

    def raw_outputs(self, board: chess.Board) -> tuple[npf32, npf32]:
        """
        NN outputs as numpy arrays
        priors numpy of shape (1880,)
        state value of shape (1,)
        """
        board_tensor = board_to_tensor(board)
        raw_policy, raw_value = self.ort_session.run(
            None, {"input": board_tensor.reshape(1, 18, 8, 8)}
        )
        return raw_policy[0], raw_value[0]

    def reset(self) -> None:
        pass
