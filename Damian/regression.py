import numpy as np
from matplotlib import pyplot as plt
import RandomFeatures as rf

#For some reason, I get errors when I try to import torch before using plt.figure()

fig = plt.figure()
ax = fig.add_subplot(111)

import torch
import torch.nn as nn
import torch.optim as optim


num_features = 2**12 + 1
num_train = 50
num_test = 100

num_features_tested = [2**k for k in range(1, 13)]
randomness = 20

func = lambda y, random, a, b: a + b * y + random * randomness
param_generator = rf.random_features(func, 2, num_features)

training_ys = np.random.normal(10, 20, num_train)
training_xs = param_generator.get_xs(training_ys)
testing_ys = np.random.normal(10, 20, num_test)
testing_xs = param_generator.get_xs(testing_ys)

training_xs = torch.tensor(training_xs, dtype = torch.float32)
training_ys = torch.tensor(training_ys, dtype = torch.float32)
testing_xs = torch.tensor(testing_xs, dtype = torch.float32)
testing_ys = torch.tensor(testing_ys, dtype = torch.float32)


baseline = nn.MSELoss()(torch.ones_like(testing_ys) * torch.mean(training_ys), testing_ys).item()

numbers_of_layers = [1, 2, 3, 4, 5]
relu_on = True

for num_layers in numbers_of_layers:
    training_losses = []
    testing_losses = []
    print("{} layers".format(num_layers))
    for features_used in num_features_tested:
        if num_layers == 1:
            model = nn.Linear(features_used, 1)
        else:
            steps = [nn.Linear(features_used, 100)]
            for _ in range(num_layers - 2):
                steps = steps + [nn.ReLU(), nn.Linear(100, 100)]
            steps = steps + [nn.ReLU(), nn.Linear(100, 1)]
            model = nn.Sequential(*steps)

        used_training_xs = training_xs[:,:features_used]
        used_testing_xs = testing_xs[:,:features_used]

        for _ in range(10000):
            opt = optim.Adam(model.parameters(), lr = 1e-3 / features_used)
            loss = nn.MSELoss()(torch.flatten(model(used_training_xs)), training_ys)
            loss.backward()
            opt.step()
            opt.zero_grad()
        training_losses.append(loss.item())
        testing_losses.append(nn.MSELoss()(torch.flatten(model(used_testing_xs)), testing_ys).item())

        print("features used: {}".format(features_used))
        print("Training loss: {}".format(training_losses[-1]))
        print("Testing loss: {}".format(testing_losses[-1]))
        print()

    ax.loglog(num_features_tested, training_losses, '-x', label = "{} layers training".format(num_layers))
    ax.loglog(num_features_tested, testing_losses, '-o', label = "{} layers testing".format(num_layers))
ax.loglog(num_features_tested, np.ones_like(num_features_tested) * baseline, label = 'baseline')
plt.legend()
plt.xlabel("Number of Features")
plt.ylabel("MSE")
plt.title("Error on Generated Data")
plt.show()




