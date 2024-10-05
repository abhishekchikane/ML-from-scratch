# # Base code
# import numpy as np
# from numpy import random
# import pdb
# random.seed(42)

# x = np.array([2,4,5,6,7])
# y = np.array([3,5,6,7,8])
# pdb.set_trace()

# # loss function
# def mse(prediction, target):
#     diff = prediction - target
#     diff_sqaure = diff**2
#     summation = sum(diff_sqaure)
#     average_loss = summation/(2*len(diff))
#     return average_loss


# weight, bias = random.randn(), random.randn()
# print(f"Randomly initialized weight and bias: {weight}, {bias}")
# learning_rate = 0.05 # tried out learning rates: 0.1, 0.01, 0.02, 0.03, 0.04

# no_of_iterations = 1000
# best_loss = np.inf

# for i in range(1, no_of_iterations):

#     # prediction
#     predictions = weight * x + bias
    
#     # calculate error
#     error = predictions - y
    
#     # calculate loss - mean squared error
#     loss = mse(predictions, y)
    
#     # Get new weights
#     new_weight = weight - learning_rate * sum(error * x)/len(x)
#     new_bias = bias - learning_rate * sum(error)/len(x)
    
#     print(f"Iteration number: {i}, loss: {loss}, new weight: {new_weight}, new bias: {new_bias}")
    
#     # store best loss and best weights
#     if best_loss > loss:
#         best_loss = loss
#         best_weight = new_weight
#         best_bias = new_bias
    
#     # update weights
#     weight = new_weight
#     bias = new_bias
    
#     # update learning rate after every 10 iterations
#     # if i%500 == 0:
#     #     learning_rate /= 2
#     #     print(f"Updating learning rate after {i} epochs. New learning rate: {learning_rate}")

# final_predictions = best_weight * x + best_bias
# print(final_predictions)

#########################################################################################################

import numpy as np
from numpy import random

class LinearRegressionUsingGradientDescent:
    def __init__(self, learning_rate=0.07, n_iterations = 100, random_state = None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weight, self.bias = None, None
        self.best_weight, self.best_bias = None, None
        self.best_loss = np.inf

    def _set_seed(self):
        if self.random_state:
            random.seed(self.random_state)
        else:
            self.random_state = 42
            random.seed(self.random_state)

    def _initialize_weight(self):
        self._set_seed()
        self.weight = random.random()
        self.bias = random.random()

    def _get_predictions_while_training(self, x):
        predictions = np.dot(x, self.weight) + self.bias
        return predictions
    
    def _get_error(self, predictions, target):
        error = predictions - target
        return error

    def _get_loss(self, error):
        error_sqaure = error **2
        error_sqaure_summation = sum(error_sqaure)
        average_error = error_sqaure_summation / (2 * len(error))
        return average_error
    
    def _update_weights(self, x, error):
        self.weight = self.weight - self.learning_rate * np.dot(x.T, error)/(len(error))
        self.bias = self.bias - self.learning_rate * sum(error)/len(error)
    
    def _get_best_weight_bias_loss(self, loss):
        if self.best_loss > loss:
            self.best_loss = loss
            self.best_weight = self.weight
            self.best_bias = self.bias

    def _update_learning_rate(self, iter):
        if iter+1 % 1000 == 0:
            self.learning_rate /= 2

    def fit(self, x, y):
        self._initialize_weight()
        for iter in range(self.n_iterations):
            predictions = self._get_predictions_while_training(x)
            error = self._get_error(predictions, y)
            loss = self._get_loss(error)
            self._update_weights(x, error)
            print(f"Iter {iter}: loss: {loss}, updated weight: {self.weight}, updated bias: {self.bias}")
            self._get_best_weight_bias_loss(loss)
            self._update_learning_rate(iter)

    def predict(self, x):
        print(f"Best parameters: {self.best_weight}, {self.best_bias}")
        predictions = np.dot(x, self.best_weight) + self.best_bias
        return predictions

x = random.randn(100)
y = random.randn(100)  
model = LinearRegressionUsingGradientDescent()
model.fit(x, y)
predictions = model.predict(x)
print(f"Target: {y[:5]}")
print(f"Predictions: {predictions[:5]}")
    