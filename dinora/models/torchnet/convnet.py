import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl


class TinyConvNet(pl.LightningModule):
    def __init__(self):
        super(TinyConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(18, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ) # 128 * 8 * 8

        self.policy = nn.Sequential(
            nn.Conv2d(8192, 0),
            nn.Flatten(),
            nn.Linear(8192, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1880),
        )

        print("Features params:", sum(p.numel() for p in self.features.parameters()))
        print("Policy params:", sum(p.numel() for p in self.policy.parameters()))


    def forward(self, x):
        x = self.features(x)
        print("Features output", x.shape)
        x = self.policy(x)
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


if __name__ == '__main__':
    import torch
    x = torch.zeros((2, 18, 8, 8))
    model = TinyConvNet()
    model(x)
