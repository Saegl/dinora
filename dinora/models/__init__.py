from dinora.models.base import BaseModel, Priors, StateValue


def model_selector(model: str) -> BaseModel:
    if model.startswith("cached_"):
        model = model.removeprefix("cached_")
        cached = True
    else:
        cached = False

    instance: BaseModel
    if model == "dnn":
        from dinora.models.dnn import DNNModel

        instance = DNNModel()

    elif model == "torchnet":
        # from dinora.models.torchnet.adapter import Torchnet
        from dinora.models.torchnet.resnet import ResNetLight

        instance = ResNetLight.load_from_checkpoint("checkpoints\models\epoch=0epoch-step=8101step.ckpt")
        
        # instance = Torchnet()
    
    elif model == "alphanet":
        import torch
        instance = torch.load('models/valid-state-5.ckpt')

    elif model == "handcrafted":
        from dinora.models.handcrafted import DummyModel

        instance = DummyModel()
    elif model == "badgyal":
        from dinora.models.badgyal import BadgyalModel

        instance = BadgyalModel()
    else:
        raise ValueError("Unknown model name")

    if cached:
        from dinora.models.cached_model import CachedModel

        return CachedModel(instance)
    else:
        return instance


__all__ = ["BaseModel", "Priors", "StateValue", "model_selector"]
