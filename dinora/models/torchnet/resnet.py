from collections import OrderedDict
import torch.nn as nn


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
            nn.Softmax(dim=-1),
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
                    ("lin2", nn.Linear(lin_channels, 1)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )
