import numpy as np
from matplotlib import pyplot as plt
import RandomFeatures as rf
from scipy.optimize import minimize_scalar
import pickle

#For some reason, I get errors when I try to import torch before using plt.figure()

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)

import torch
import torch.nn as nn
import torch.optim as optim


num_features = 2**12 + 1
num_train = 50
num_test = 100

num_features_tested = [2**k for k in range(1,5)]# 13)]
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


baseline = nn.MSELoss()(torch.ones_like(testing_ys) * torch.mean(training_ys), testing_ys).item()

numbers_of_layers = [1, 2]#, 3, 4, 5]
colors_for_layers = ['red','orange']#,'yellow','green','blue']
relu_on = False

compare_bayesian = True

if compare_bayesian:
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

    ax.loglog(num_features_tested, training_losses, '--xk', label = "Bayesian training")
    ax.loglog(num_features_tested, testing_losses, '--ok', label = "Bayesian testing")
    
for num_layers, my_color in zip(numbers_of_layers, colors_for_layers):
    training_losses = []
    testing_losses = []
    print("{} layers".format(num_layers))
    for features_used in num_features_tested:
        if num_layers == 1:
            model = nn.Linear(features_used, 1)
        else:
            steps = [nn.Linear(features_used, 100)]
            for _ in range(num_layers - 2):
                if relu_on:
                    steps = steps + [nn.ReLU(), nn.Linear(100, 100)]
                else:
                    steps = steps + [nn.Linear(100, 100)]
            if relu_on:
                steps = steps + [nn.ReLU(), nn.Linear(100, 1)]
            else:
                steps = steps + [nn.Linear(100, 1)]
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

        # pickle.dump(model, open('model'+str(num_layers)+'.pkl', 'wb'))

        print("features used: {}".format(features_used))
        print("Training loss: {}".format(training_losses[-1]))
        print("Testing loss: {}".format(testing_losses[-1]))
        print(my_color)

    ax.loglog(num_features_tested, training_losses, color=my_color, marker='x', label = "{} layers training".format(num_layers))
    ax.loglog(num_features_tested, testing_losses, color=my_color, marker='o', label = "{} layers testing".format(num_layers))



ax.loglog(num_features_tested, np.ones_like(num_features_tested) * baseline, '-.', label = 'baseline')
plt.legend()
plt.xlabel("Number of Features")
plt.ylabel("MSE")
if relu_on:
    plt.title("Error on Generated Data")
else:
    plt.title("Error on Generated Data without ReLU")
plt.show()