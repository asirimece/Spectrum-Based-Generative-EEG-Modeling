from torch import optim
from torch.optim import Optimizer
from lib.gen import OptimConfig


def get_optimizer(model, config: OptimConfig) -> Optimizer:
    match config.name:
        case 'adam':
            return optim.Adam(model.parameters(), lr=config.lr, **config.kwargs)
        case 'sgd':
            return optim.SGD(model.parameters(), lr=config.lr, **config.kwargs)
        case 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=config.lr, **config.kwargs)
        case _:
            raise ValueError(f'Unknown optimizer: {config.name}')
