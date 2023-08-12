import math
import chess
import torch

from dinora.models.torchnet.resnet import ResNetLight
from dinora.board_representation import board_to_tensor
from dinora.policy import extract_prob_from_policy
from dinora.models import BaseModel, Priors, StateValue


class Torchnet(BaseModel):
    def __init__(self) -> None:
        # self.model = ResNet(
        #     res_channels = 64,
        #     res_blocks = 8,
        #     policy_channels = 64,
        #     value_channels = 16,
        #     value_lin_channels = 64,
        # ).to("cuda")
        self.model = ResNetLight.load_from_checkpoint(
            "checkpoints\models\epoch=0epoch-step=8101step.ckpt"
        )
        # self.model.load_state_dict(torch.load("models/model-1epoch-75chunk.pth"))
        self.model.eval()
        self.softmax_temp = 1.6

    def model_out(self, board: chess.Board):
        board_tensor = board_to_tensor(board)

        with torch.no_grad():
            pred = self.model(
                torch.from_numpy(board_tensor).reshape((1, 18, 8, 8)).to("cuda")
            )

        return pred

    def raw_eval(self, board: chess.Board) -> tuple[Priors, StateValue]:
        policy, value = self.model_out(board)
        # unwrap tensor
        policy = policy[0]
        value = float(value[0]) * 2.0 - 1.0

        # take only legal moves from policy
        t = self.softmax_temp
        moves = []
        policies = []

        def lookup(policy, move):
            return extract_prob_from_policy(policy, move, not board.turn)

        for move in board.legal_moves:
            move_prior = lookup(policy, move)
            moves.append(move)
            policies.append(math.exp(move_prior.item() / t))

        # map to sum(policies) == 1
        s = sum(policies)
        policies_map = map(lambda e: math.exp(e / t) / s, policies)

        return dict(zip(moves, policies_map)), value

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        result = board.result(claim_draw=True)
        if result == "*":
            # Game is not ended
            # evaluate by using ANN
            priors, value_estimate = self.raw_eval(board)
        elif result == "1/2-1/2":
            # It's already draw
            # or we can claim draw, anyway `value_estimate` is 0.0
            # TODO: should I set priors = {}?
            # It's logical to set it empty because there is no need
            # to calculate deeper already draw position,
            # but with low time/nodes search, it leads to
            # empty node.children bug
            priors, _ = self.raw_eval(board)
            value_estimate = 0.0
        else:
            # result == '1-0' or result == '0-1'
            # we are checkmated because it's our turn to move
            # so the `value_estimate` is -1.0
            priors = {}  # no moves after checkmate
            value_estimate = -1.0
        return priors, value_estimate
