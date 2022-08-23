# Implement multi layer perceptron

import numpy as np

class MLP:
    def __init__(self, nInputs=3, nHidden=[3,4,5], nOutputs=2):
        self.nInputs = nInputs
        self.nHidden = nHidden
        self.nOutputs = nOutputs
        
        layers = [self.nInputs] + self.nHidden + [self.nOutputs]
        # init random weights
        self.weights = []
        
        # loops through each "connection" between two layers and generates a random weight with the
        # necessary dimensions. ie. one for inputs to 1st layer, next for 1st layer to 2nd layer. and so on
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)
        
            
    def forwardProp(self, inputs):
        # first layer the activation values are the inputs
        activations = inputs
        
        for w in self.weights:
            # caluclate net inputs (h)
            # multiply outputs of previous layer with the current weightings
            netInputs = np.dot(activations, w)
            
            # then calculate activation values (using sigmoid)
            activations = self._sigmoid(netInputs)

        return activations
        
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    


if __name__ == "__main__":
    # instantiate MLP class
    mlp = MLP()
    
    # init some inputs
    inputs = np.random.rand(mlp.nInputs)
    
    # preform forward propagation
    outputs = mlp.forwardProp(inputs)
    
    # print output
    print("Input is: {}".format(inputs))
    print("Output is: {}".format(outputs))