from dinora.models.alphanet import AlphaNet

model = AlphaNet(
    filters=128,
    res_blocks=5,
    policy_channels=8,
    value_channels=8,
    value_fc_hidden=32,
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"\nTotal parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
