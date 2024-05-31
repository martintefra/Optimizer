# optimizers/optimizer_factory.py
import torch.optim as optim

from optimizers.SignSGD import SignSGD

def get_optimizer(name, model_parameters, lr=0.001, momentum=0, dampening=0, weight_decay=0):
    if name == 'SGD':
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
    elif name == 'Adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'SignSGD':
        return SignSGD(model_parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'Adagrad':
        return optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
