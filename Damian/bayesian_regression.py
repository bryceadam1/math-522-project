import numpy as np
from matplotlib import pyplot as plt
import RandomFeatures as rf
from scipy.optimize import minimize_scalar

#For some reason, I get errors when I try to import torch before using plt.figure()

fig = plt.figure()
ax = fig.add_subplot(111)

import torch
import torch.nn as nn
import torch.optim as optim


num_features = 2**12 + 1
num_train = 50
num_test = 100

num_features_tested = [2**k for k in range(1, 5)]
randomness = 20

func = lambda y, random, a, b: a + b * y + random * randomness
param_generator = rf.random_features(func, 2, num_features)

training_ys_array = np.random.normal(10, 20, num_train)
training_xs_array = param_generator.get_xs(training_ys_array)
testing_ys_array = np.random.normal(10, 20, num_test)
testing_xs_array = param_generator.get_xs(testing_ys_array)

training_xs = torch.tensor(training_xs_array, dtype = torch.float32)
training_ys = torch.tensor(training_ys_array, dtype = torch.float32)
testing_xs = torch.tensor(testing_xs_array, dtype = torch.float32)
testing_ys = torch.tensor(testing_ys_array, dtype = torch.float32)

training_losses = []
testing_losses = []

for features_used in num_features_tested:
    #inverse of the covariance matrix for a and b for each feature used
    sig_invs = [np.matrix([[1., 0.], [0., 1.]]) for  _ in range(features_used)]
    #what we expect a and b to be on average
    means = [np.array([[0.], [0.]]) for _ in range(features_used)]

    #Update to find what we expect alpha and beta to be
    for j in range(num_train):
        y = training_ys_array[j]
        for i in range(features_used):
            x = training_xs_array[j, i]

            sig_inv = sig_invs[i]
            mean = means[i]

            sig_inv_2 = np.array([[1 / randomness**2, y / randomness**2], [y / randomness**2, y**2 / randomness**2]])
            mean_2 = np.array([[x], [0]])

            sig_inv_new = sig_inv + sig_inv_2
            mean_new = np.linalg.solve(sig_inv_new, sig_inv @ mean + sig_inv_2 @ mean_2)

            sig_invs[i] = sig_inv_new

            means[i] = mean_new

    sigs = [np.linalg.inv(inv) for inv in sig_invs]
    
    predictions = []
    for j in range(num_test):
        #Finds the maximum a posteriori
        #We know the distributions of 

        def neg_log_likelihood(y):
            mu_y = 10
            var_y = 400
            result = -np.log(1 / np.sqrt(2 * np.pi * var_y)) + 1 / 2 * (y - mu_y)**2 / var_y
            for i in range(features_used):
                mu_x = means[i][0,0] + means[i][1,0] * y
                var_x = sigs[i][0, 0] + 2 * y * sigs[i][0, 1] + y**2 * sigs[i][1,1] + randomness**2

                x = testing_xs_array[j, i]
                result += -np.log(1 / np.sqrt(2 * np.pi * var_x)) + 1 / 2 * (x - mu_x)**2 / var_x

            return result
        predictions.append(minimize_scalar(neg_log_likelihood).x)
    predictions = torch.tensor(predictions)
    loss = nn.MSELoss()(predictions, testing_ys).item()
    testing_losses.append(loss)

    training_predictions = []
    for j in range(num_train):
        #Finds the maximum a posteriori
        #We know the distributions of 

        def neg_log_likelihood(y):
            mu_y = 10
            var_y = 400
            result = -np.log(1 / np.sqrt(2 * np.pi * var_y)) + 1 / 2 * (y - mu_y)**2 / var_y
            for i in range(features_used):
                mu_x = means[i][0,0] + means[i][1,0] * y
                var_x = sigs[i][0, 0] + 2 * y * sigs[i][0, 1] + y**2 * sigs[i][1,1] + randomness**2

                x = training_xs_array[j, i]
                result += -np.log(1 / np.sqrt(2 * np.pi * var_x)) + 1 / 2 * (x - mu_x)**2 / var_x

            return result
        training_predictions.append(minimize_scalar(neg_log_likelihood).x)
    training_predictions = torch.tensor(training_predictions)
    loss = nn.MSELoss()(training_predictions, training_ys).item()
    training_losses.append(loss)

    print("features used: {}".format(features_used))
    print("loss: {}".format(loss))

ax.loglog(num_features_tested, testing_losses, label = "Bayesian testing")
ax.loglog(num_features_tested, training_losses, label = "Bayesian training")


plt.legend()
plt.show()