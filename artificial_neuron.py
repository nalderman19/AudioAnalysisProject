"""
Small script to demonstrate an implementation of an artificial neuron

Main function runs a set of inputs and their weights through a
sigmoid activation function.
"""
import math

def sigmoid(h):
	# summation of inputs and their weights is passed through the
	# sigmoid function
	y = 1.0 / (1 + math.exp(-h))

	return y

def activate(inputs, weights):
	# the inputs and weights are zipped together for easy and quick
	# summation of h
	h = 0
	for k,w in zip(inputs, weights):
		h += k*w

	# activation returns sigmoid result of h
	return sigmoid(h)

if __name__ == "__main__":
	inputs = [.5,.3,.2]
	weights = [.4,.7,.2]
	y = activate(inputs, weights)
	print(y)