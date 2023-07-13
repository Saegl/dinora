import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

class LinearNN(pl.LightningModule):
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
        )
        self.value_output = nn.Sequential(
            nn.Linear(2048, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return self.policy_output(logits), self.value_output(logits)

    def training_step(self, batch, batch_idx):
        x, (y_policy, y_value) = batch
        y_hat_policy, y_hat_value = self(x)
        
        policy_loss = F.cross_entropy(y_hat_policy, y_policy)
        value_loss = F.cross_entropy(y_hat_value, y_value)
        cumulative_loss = policy_loss + value_loss

        self.log_dict({
            'train/policy_loss': policy_loss,
            'train/value_loss': value_loss,
            'train/cumulative_loss': cumulative_loss
        })
        
        return cumulative_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
