# Implement multi layer perceptron

import numpy as np
from random import random

# save activations and derivatives in __init__ class method
# implement backprop
# implement gradient descent
# implement train method
# train out net with dataset
# make predictions

class MLP:
    def __init__(self, nInputs=3, nHidden=[3,4,5], nOutputs=2):
        self.nInputs = nInputs
        self.nHidden = nHidden
        self.nOutputs = nOutputs
        
        layers = [self.nInputs] + self.nHidden + [self.nOutputs]
        # init random weights
        self.weights = []
        
        # loops through each "connection" between two layers and generates a random weight matrix with the
        # necessary dimensions. ie. one for inputs to 1st layer, next for 1st layer to 2nd layer. and so on
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)
            
        # init activation matrix
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        # init derivatives matrix
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros(layers[i])
            derivatives.append(d)
        self.derivatives = derivatives
            
        
    def forwardProp(self, inputs):
        # first layer the activation values are the inputs
        activations = inputs
        self.activations[0] = activations
        
        for i,w in enumerate(self.weights):
            # caluclate net inputs (h)
            # multiply outputs of previous layer with the current weightings
            netInputs = np.dot(activations, w)
            
            # then calculate activation values (using sigmoid)
            activations = self._sigmoid(netInputs)
            self.activations[i+1] = activations

        return activations
    
    
    def backProp(self, error):
        
        # from comp325
        # dE/dW_i = (y - a_[i+1] * s'(h_[i+1])) * a_i
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]
        
        
        for i in reversed(range(len(self.derivatives))):
            # need to get activation of next i
            activations = self.activations[i+1] # this is the third line of equations
            
            
            delta = error * self._sigmoid_derivative(activations) # this is the first line of equations
            delta_reshape = delta.reshape(delta.shape[0],-1).T             
            
            currentActivations = self.activations[i] # ndarray([1,2]) --> ndarray([[1],[2]])
            currentActivations_reshape = currentActivations.reshape(currentActivations.shape[0], -1)
                        
            self.derivatives[i] = np.dot(currentActivations_reshape, delta_reshape)
            
            # set up error for next loop
            error = np.dot(delta, self.weights[i].T)
        
        return error
        
    
    def gradientDescent(self, lr):
        # loop through all weights, 
        for i in range(len(self.weights)): # update weights with their derivitaves and learning rate
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            
            weights += derivatives * lr
    
        return weights


    def train(self, inputs, targets, epochs, lr):
        
        for i in range(epochs):
            sumError = 0
            for input, target in zip(inputs, targets): # get inputs and targets one by one
                # preform forward propagation
                output = self.forwardProp(input)
                
                # calculate error
                error = target - output
                
                # backprop
                self.backProp(error)
                
                # gradient descent
                self.gradientDescent(lr)
                
                sumError += self._mse(target, output)
                
            # report error
            #print("Error: {} at epoch {}".format(sumError / len(inputs), i+1))

    def _mse(self, target, output): # mean squared error
        return np.average((target - output)**2)

        
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
        
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    


if __name__ == "__main__":
    
    # made up dataset to aid computing sum of inputs... see structure in spyder variable explorer
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    
    # instantiate MLP class
    mlp = MLP(2, [5], 1)
    
    # train (update weights based off of gradient descent)
    mlp.train(inputs, targets, 50, 0.3)
    
    # create dummy data to make predictions with
    input = np.array([0.3,0.1])
    target = np.array([0.4])
    
    output = mlp.forwardProp(input)
    
    print("The sum of {} and {} is {}".format(input[0],input[1],output[0]))
    