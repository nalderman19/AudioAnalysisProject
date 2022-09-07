# Implement multi layer perceptron

import numpy as np

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
            
            weights = weights + derivatives * learning rate
    
    return weights
        
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
        
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    


if __name__ == "__main__":
    # instantiate MLP class
    mlp = MLP(2, [5], 1)
    
    # init some inputs & target
    inputs = np.array([0.1,0.2])
    target = np.array([0.3])
    
    # preform forward propagation
    outputs = mlp.forwardProp(inputs)
    
    # calculate error
    error = target - outputs
    
    # backprop
    mlp.backProp(error)
    
    # gradient descent
    mlp.gradientDescent(0.1)
    
    # train (update weights based off of gradient descent)
    
    # print output
    print("Input is: {}".format(inputs))
    print("Output is: {}".format(outputs))