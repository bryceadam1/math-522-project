#from matplotlib import pyplot as plt

#plt.figure().add_subplot().plot([2,3,4])

import torch
import torch.nn as nn
import torch.optim as optim


def combine_linear_layers(model):
    if type(model) is nn.Linear:
        return model.weight.detach().numpy().T
    result = None
    for layer in model:
        print(layer)
        M = layer.weight.detach().numpy().T
        if result is None:
            result = M
        else:
            result = result @ M
    return result




model = nn.Sequential(nn.Linear(10, 100), nn.Linear(100, 2), nn.Linear(2, 1))
nn.Linear(10, 100).weight.detach().numpy().T
print(combine_linear_layers(model))