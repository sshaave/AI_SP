import numpy as np

# Passed-ion gradient from next layer
# for the purpose of this example we're going to use
# a vector of 1s, 3 derivatives - one for each neuron
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

biases = np.array([[2, 3, 0.5]])
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# Sum weights related to the given input multiplied by
# the gradient related to the given neuron
#dx0 = sum(weights[0]*dvalues[0])
#dx1 = sum(weights[1]*dvalues[0])
#dx2 = sum(weights[2]*dvalues[0])
#dx3 = sum(weights[3]*dvalues[0])

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

dweights = np.dot(inputs.T, dvalues)
#dinputs = np.array([dx0, dx1, dx2, dx3])
dinputs = np.dot(dvalues, weights.T)
print(dinputs)
print(dweights)