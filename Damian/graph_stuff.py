import RandomFeatures as rf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

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


training_losses = []
testing_losses = []
baseline = nn.MSELoss()(torch.ones_like(testing_ys) * torch.mean(training_ys), testing_ys).item()


for features_used in num_features_tested:
    model = nn.Linear(features_used, 1)

    used_training_xs = training_xs[:,:features_used]
    used_testing_xs = testing_xs[:,:features_used]

    for _ in range(10000):
        #opt = optim.Adam(model.parameters(), lr = 1e-3 / features_used)
        opt = optim.Adam(model.parameters(), lr = 1e-3 / features_used)
        #opt = optim.SGD(model.parameters(), lr = 1e-4)
        loss = nn.MSELoss()(torch.flatten(model(used_training_xs)), training_ys)
        loss.backward()
        opt.step()
        opt.zero_grad()
    training_losses.append(loss.item())
    testing_losses.append(nn.MSELoss()(torch.flatten(model(used_testing_xs)), testing_ys).item())

    print(features_used)
    print("Training loss", training_losses[-1])
    print("Testing loss", testing_losses[-1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(num_features_tested, training_losses)
ax.loglog(num_features_tested, testing_losses)
ax.loglog(num_features_tested, np.ones_like(num_features_tested) * baseline)
plt.legend(["Training loss", "Testing loss", "Baseline"])
plt.show()




