import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

from dinora.dataset2 import download_ccrl_dataset, random_dataset
from dinora.models.torchnet.linear_net import LinearNN
from dinora.models.torchnet.resnet import ResNet


if not torch.cuda.is_available():
    raise Exception("Cuda is not available!")

device = "cuda"
batch_size = 2048
learning_rate = 1e-3
epochs = 10
log_freq = 50  # batches
checkpoint_freq = 4  # chunks
chunks_count = 250  # 250 max

# train_chunks, test_chunks = random_dataset()
# dataset_name = 'Random'

train_chunks, test_chunks = download_ccrl_dataset(chunks_count=chunks_count)
dataset_name = f"CCRL-{chunks_count}chunks"
# 250 chunks
# each chunk = 10k games => 2.5 million games
# each game ~=~ 40 moves ~=~ 80 positions => 200 million positions

# in one chunk there is 10k * 80 positions = 800k positions

res_channels = 64
res_blocks = 8
policy_channels = 64
value_channels = 16
value_lin_channels = 64
model = ResNet(
    res_channels,
    res_blocks,
    policy_channels,
    value_channels,
    value_lin_channels,
).to(device)
# model = LinearNN().to(device)

policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

run = wandb.init(
    project="dinora-chess",
    # mode="disabled",
    config={
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "logging_freq": log_freq,
        "checkpoint_freq": checkpoint_freq,
        "optimizer": optimizer.__class__.__name__,
        "dataset": dataset_name,
        "model_class": model.__class__.__name__,
        "res_channels": res_channels,
        "res_blocks": res_blocks,
        "policy_channels": policy_channels,
        "value_channels": value_channels,
        "value_lin_channels": value_lin_channels,
    },
)


def checkpoint(checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)
    wandb.save(checkpoint_name)
    print(f"Model saved as {checkpoint_name}")


def train(
    dataloader: DataLoader,
    model: nn.Module,
    policy_loss_fn: nn.CrossEntropyLoss,
    value_loss_fn: nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    logging_frequency: int,
):
    print("Start train on chunk")
    model.train()
    for batch, (X, (y_policy, y_value)) in tqdm(enumerate(dataloader)):
        X, (y_policy, y_value) = X.to(device), (y_policy.to(device), y_value.to(device))

        # Compute prediction error
        policy_pred, value_pred = model(X)
        policy_loss = policy_loss_fn(policy_pred, y_policy)
        value_loss = value_loss_fn(value_pred, y_value)

        # Backpropagation
        optimizer.zero_grad()
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        if batch % logging_frequency == 0:
            policy_loss = policy_loss.item()
            value_loss = value_loss.item()
            current = batch * len(X)
            print(
                f"\n\n policy_loss: {policy_loss:>7f} value_loss: {value_loss:>7f}"
                f"\n [states: {current:>5d}]\n"
            )
            wandb.log(
                {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "states": current,
                }
            )


def test(
    dataloader,
    model,
    policy_loss_fn: nn.CrossEntropyLoss,
    value_loss_fn: nn.MSELoss,
):
    print("Start test on chunk")
    model.eval()
    policy_loss = 0.0
    value_loss = 0.0
    policy_correct = 0

    with torch.no_grad():
        num = 0
        for num, (X, (y_policy, y_value)) in tqdm(enumerate(dataloader)):
            X, (y_policy, y_value) = X.to(device), (
                y_policy.to(device),
                y_value.to(device),
            )

            policy_pred, value_pred = model(X)
            policy_loss += policy_loss_fn(policy_pred, y_policy).item()
            value_loss += value_loss_fn(value_pred, y_value).item()
            policy_correct += (
                (policy_pred.argmax(1) == y_policy.argmax(1))
                .type(torch.float)
                .sum()
                .item()
            )

    num_batches = num + 1
    size = num_batches * batch_size

    policy_loss /= num_batches
    policy_correct /= size
    value_loss /= num_batches

    test_policy_accuracy = 100 * policy_correct
    test_policy_loss = policy_loss
    test_value_loss = value_loss

    print(
        f"Test Error: \n Policy accuracy: {(100*policy_correct):>0.1f}%, Policy loss: {policy_loss:>8f}, Value loss: {value_loss:>8f} \n"
    )
    wandb.log(
        {
            "test_policy_accuracy": test_policy_accuracy,
            "test_policy_loss": test_policy_loss,
            "test_value_loss": test_value_loss,
        }
    )


chunk_count = 0
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    for chunk_idx in range(len(train_chunks)):
        print(f"Chunk {chunk_idx + 1}\n-------------------------")

        train_dataloader = DataLoader(train_chunks[chunk_idx], batch_size=batch_size)
        test_dataloader = DataLoader(test_chunks[chunk_idx], batch_size=batch_size)

        train(
            train_dataloader,
            model,
            policy_loss_fn,
            value_loss_fn,
            optimizer,
            log_freq,
        )
        test(test_dataloader, model, policy_loss_fn, value_loss_fn)

        chunk_count += 1
        if chunk_count % checkpoint_freq == 0:
            checkpoint(f"checkpoints/model-{epoch+1}epoch-{chunk_count}chunk.pth")

checkpoint(f"checkpoints/model-final.pth")
