import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    name: cls
    for name, cls in vars(optim).items()
    if (
        isinstance(cls, type)
        and issubclass(cls, Optimizer)
        and cls is not Optimizer
        and not name.startswith("_")
    )
}

SCHEDULERS: dict[str, type[LRScheduler]] = {
    name: cls
    for name, cls in vars(lr_sched).items()
    if (
        isinstance(cls, type)
        and issubclass(cls, LRScheduler)
        and cls is not LRScheduler
        and not name.startswith("_")
    )
}
