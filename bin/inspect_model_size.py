from lightning.pytorch.utilities.model_summary.model_summary import summarize

from dinora.models.alphanet import AlphaNet

model = AlphaNet(
    filters=128,
    res_blocks=5,
    policy_channels=8,
    value_channels=8,
    value_fc_hidden=32,
)
print(summarize(model, max_depth=10))
