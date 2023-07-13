import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl


class TinyConvNet(pl.LightningModule):
    def __init__(self, policy_moves):
        super(TinyConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(18, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, policy_moves),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, (y_policy, y_value) = batch
        y_hat_policy = self(x)

        policy_loss = F.cross_entropy(y_hat_policy, y_policy)

        self.log_dict({
            'train/policy_loss': policy_loss,
        })

        return policy_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def on_fit_end(self):
        metrics = self.trainer.callback_metrics
        print(metrics)
