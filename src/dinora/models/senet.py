from typing import Any

import lightning.pytorch as pl
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from dinora.models.alphanet import AlphaNet
from dinora.models.base import BaseModel

npf32 = npt.NDArray[np.float32]


class ChannelAttentionModule(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 2) -> None:
        """
        Channel-wise attention module, Squeeze-and-Excitation Networks Jie Hu1, Li Shen, Gang Sun - https://arxiv.org/pdf/1709.01507v2.pdf
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class ResBlockSE(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=filters),
        )
        self.relu = nn.ReLU()
        self.se = ChannelAttentionModule(channels=filters)

    def forward(self, x):  # type: ignore
        out = self.body(x)
        out = self.se(out)
        return self.relu(x + out)


class SeNet(AlphaNet):
    def __init__(
        self,
        filters: int = 256,
        res_blocks: int = 19,
        policy_channels: int = 64,
        value_channels: int = 8,
        value_fc_hidden: int = 256,
        value_loss_weight: float = 0.1,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        optimizer_params: dict[str, Any] | None = None,
        scheduler_name: str = "StepLR",
        scheduler_params: dict[str, Any] | None = None,
        scheduler_frequency: int = 1000,
    ):
        # Call __init__ on parents except AlphaNet
        # because we are substituting ResBlock with ResBlockSE
        pl.LightningModule.__init__(self)
        BaseModel.__init__(self)

        self.value_loss_weight = value_loss_weight
        self.learning_rate = learning_rate
        if optimizer_params is None:
            raise ValueError("optimizer_params is None")

        if scheduler_params is None:
            raise ValueError("scheduler_params is None")

        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.scheduler_frequency = scheduler_frequency

        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=18,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *(ResBlockSE(filters) for _ in range(res_blocks))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=policy_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=policy_channels * 8 * 8, out_features=1880),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=value_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=value_channels * 8 * 8, out_features=value_fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=value_fc_hidden, out_features=1),
            nn.Tanh(),
        )
