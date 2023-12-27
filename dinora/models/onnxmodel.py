import chess
import numpy as np
import onnxruntime

from dinora.models.nnwrapper import NNWrapper
from dinora.encoders.board_representation import board_to_tensor


def softmax(x, tau=1.0):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


class OnnxModel(NNWrapper):
    def __init__(self, weights, device: str):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device '{device}' is not supported")

        self.ort_session = onnxruntime.InferenceSession(weights, providers=providers)

    def raw_outputs(self, board: chess.Board):
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

    def reset(self):
        pass
