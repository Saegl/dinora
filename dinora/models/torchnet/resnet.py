import math
from collections import OrderedDict

import chess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import StepLR

import lightning.pytorch as pl

from dinora.board_representation2 import board_to_tensor
from dinora.policy2 import extract_prob_from_policy


class ResNet(nn.Module):
    def __init__(
        self,
        res_channels: int,
        res_blocks: int,
        policy_channels: int,
        value_channels: int,
        value_lin_channels: int,
    ):
        super().__init__()
        self.conv_block = ConvBlock(18, res_channels, 3, padding=1)

        blocks = [(f"resblock{i}", (ResBlock(res_channels))) for i in range(res_blocks)]
        self.res_stack = nn.Sequential(OrderedDict(blocks))

        self.policy_head = PolicyHead(res_channels, policy_channels)
        self.value_head = ValueHead(res_channels, value_channels, value_lin_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_stack(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                    ("bn1", nn.BatchNorm2d(channels)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("conv2", nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                    ("bn2", nn.BatchNorm2d(channels)),
                ]
            )
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x

        x = self.layers(x)

        x = x + x_in
        x = self.relu2(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, in_channels, policy_channels):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, policy_channels, 1),
            Flatten(),
            nn.Linear(8 * 8 * policy_channels, 1880),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        return x.view(x.size(0), -1)


class ValueHead(nn.Sequential):
    def __init__(self, in_channels, value_channels, lin_channels):
        super().__init__(
            OrderedDict(
                [
                    ("conv_block", ConvBlock(in_channels, value_channels, 1)),
                    ("flatten", Flatten()),
                    ("lin1", nn.Linear(value_channels * 8 * 8, lin_channels)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("lin2", nn.Linear(lin_channels, 3)),
                ]
            )
        )


class ResNetLight(pl.LightningModule):
    def __init__(
        self,
        res_channels: int,
        res_blocks: int,
        policy_channels: int,
        value_channels: int,
        value_lin_channels: int,
        learning_rate: float,
        lr_scheduler_gamma: float,
        lr_scheduler_freq: int,
    ):
        super().__init__()
        self.hparams.learning_rate = learning_rate
        self.save_hyperparameters()
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.lr_scheduler_freq = lr_scheduler_freq

        self.conv_block = ConvBlock(18, res_channels, 3, padding=1)

        blocks = [(f"resblock{i}", (ResBlock(res_channels))) for i in range(res_blocks)]
        self.res_stack = nn.Sequential(OrderedDict(blocks))

        self.policy_head = PolicyHead(res_channels, policy_channels)
        self.value_head = ValueHead(res_channels, value_channels, value_lin_channels)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)
        x = self.res_stack(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    def training_step(self, batch, batch_idx):
        x, (y_policy, y_value) = batch
        batch_len = len(x)

        y_hat_policy, y_hat_value = self(x)
        
        policy_loss = F.cross_entropy(y_hat_policy, y_policy)
        value_loss = F.cross_entropy(y_hat_value, y_value)
        cumulative_loss = policy_loss + value_loss

        policy_accuracy = (
            (y_hat_policy.argmax(1) == y_policy)
            .float()
            .sum()
            .item()
        ) / batch_len

        value_accuracy = (
            (y_hat_value.argmax(1) == y_value)
            .float()
            .sum()
            .item()
        ) / batch_len

        self.log_dict({
            'train/policy_accuracy': policy_accuracy,
            'train/policy_loss': policy_loss,
            'train/value_accuracy': value_accuracy,
            'train/value_loss': value_loss,
            'train/cumulative_loss': cumulative_loss
        })
        
        return cumulative_loss
    
    def validation_step(self, batch: tuple[Tensor, tuple[Tensor, Tensor]], batch_idx):
        x, (y_policy, y_value) = batch
        batch_len = len(x)

        y_hat_policy: torch.Tensor
        y_hat_value: torch.Tensor
        y_hat_policy, y_hat_value = self(x)
        
        policy_accuracy = (
            (y_hat_policy.argmax(1) == y_policy)
            .float()
            .sum()
            .item()
        ) / batch_len

        outcomes_matches = y_hat_value.argmax(1) == y_value

        value_accuracy = outcomes_matches.float().sum().item() / batch_len

        win_mask = y_value == 0
        win_count = win_mask.type(torch.float).sum().item()
        if win_count != 0:
            win_accuracy = (outcomes_matches * win_mask) \
                .type(torch.float).sum().item() / win_count
            self.log("validation/win_accuracy", win_accuracy)
            
        
        draw_mask = y_value == 1
        draw_count = draw_mask.type(torch.float).sum().item()
        if draw_count != 0:
            draw_accuracy = (outcomes_matches * draw_mask) \
                .type(torch.float).sum().item() / draw_count
            self.log('validation/draw_accuracy', draw_accuracy)
        
        lose_mask = y_value == 2
        lose_count = lose_mask.type(torch.float).sum().item()
        if lose_count != 0:
            lose_accuracy = (outcomes_matches * lose_mask) \
                .type(torch.float).sum().item() / lose_count
            self.log("validation/lose_accuracy", lose_accuracy)

        self.log_dict({
            "validation/policy_accuracy": policy_accuracy,
            "validation/policy_loss": F.cross_entropy(y_hat_policy, y_policy).item(),
            "validation/value_accuracy": value_accuracy,
            "validation/value_loss": F.cross_entropy(y_hat_value, y_value).item(),
        })

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(
            optimizer,
            step_size=1,
            gamma=self.lr_scheduler_gamma,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': self.lr_scheduler_freq,
            },
        }
    
    def eval_by_network(self, board: chess.Board):
        board_tensor = board_to_tensor(board)

        with torch.no_grad():
            raw_policy, raw_value = self(
                torch.from_numpy(board_tensor).reshape((1, 18, 8, 8)).to(self.device)
            )
        
        # TODO: Refactor this func, write softmax with temp
        unwrapped_raw_policy = F.softmax(raw_policy[0])
        outcomes_probs = F.softmax(raw_value[0], dim=-1)

        # take only legal moves from policy
        t = 2.72
        moves = []
        policies = []
        lookup = lambda policy, move: extract_prob_from_policy(policy, move, not board.turn)
        for move in board.legal_moves:
            move_prior = lookup(unwrapped_raw_policy, move)
            moves.append(move)
            policies.append(math.exp(move_prior.item() / t))

        # map to sum(policies) == 1
        s = sum(policies)
        policies_map = map(lambda e: math.exp(e / t) / s, policies)

        return dict(zip(moves, policies_map)), outcomes_probs

    def evaluate(self, board: chess.Board):
        result = board.result(claim_draw=True)
        if result == "*":
            # Game is not ended
            # evaluate by using ANN
            priors, value_estimate = self.eval_by_network(board)
            value_estimate = (value_estimate[0] - value_estimate[2]).item()
        elif result == "1/2-1/2":
            # It's already draw
            # or we can claim draw, anyway `value_estimate` is 0.0
            # TODO: should I set priors = {}?
            # It's logical to set it empty because there is no need
            # to calculate deeper already draw position,
            # but with low time/nodes search, it leads to
            # empty node.children bug
            priors, _ = self.eval_by_network(board)
            value_estimate = 0.0
        else:
            # result == '1-0' or result == '0-1'
            # we are checkmated because it's our turn to move
            # so the `value_estimate` is -1.0
            priors = {}  # no moves after checkmate
            value_estimate = -1.0
        return priors, value_estimate
