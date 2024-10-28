'''
Basic things:
1) dataset
2) Hyperparameters - no of iterations, learning rate
3) random weights, bias
4) best weights, bias, loss
5) loss function
6) training loop
    predictions
    error
    loss
    update weights
    update best loss
'''

import numpy as np
from numpy import random
random.seed(42)

# dataset
x = random.randn(100,3)
y = random.randn(100)

# Initialize weights
weights = random.randn(x.shape[1])
bias = random.randn()

# hyperparameters
n_iterations, learning_rate = 100, 0.5

# best params
best_weights, best_bias, best_loss = None, None, np.inf

# Define loss function
def loss_function(error):
    error_square = error ** 2
    summation = sum(error_square)
    average_error = summation/(2*len(error))
    return average_error

# Model training
for i in range(n_iterations):
    y_pred = np.dot(x, weights) + bias
    error = y_pred - y
    loss = loss_function(error)
    weights = weights - learning_rate * np.dot(x.T, error) / len(error)
    bias = bias - learning_rate * sum(error) / len(error)
    if loss < best_loss:
        best_loss = loss
        best_weights = weights
        best_bias = bias
    print(i, best_loss, best_weights, best_bias)

y_predictions = np.dot(x, best_weights) + best_bias

print(y[:10])
print(y_predictions[:10])