from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
        )

        self.policy_output = nn.Sequential(
            nn.Linear(2048, 1880),
            nn.Softmax(dim=-1),
        )
        self.value_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return self.policy_output(logits), self.value_output(logits)
