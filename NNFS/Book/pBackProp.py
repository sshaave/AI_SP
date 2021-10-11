import numpy as np

#Forward pass
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function
y = max(z, 0)

# Backward pass

# The derivative from the next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)

# Partial derivatives of the sum, the chain rule
dsum_dwx0 = 1
dsum_dwx1 = 1
dsum_dwx2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dwx0
drelu_dxw1 = drelu_dz * dsum_dwx1
drelu_dxw2 = drelu_dz * dsum_dwx2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[0]
#drelu_dx0 = drelu_dxw0 * dmul_dx0
#print(drelu_dx0)

drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
#print(drelu_dx0)

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s

dvalues = np.array([[1., 1., 1.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T


# Sum weights related to the given input multiplied bu
# the gradient related to the given neuron
dx0 = sum([weights[0][0] *dvalues[0][0], weights[0][1]*dvalues[0][1],
          weights[0][2]*dvalues[0][2]])
dx1 = sum([weights[1][0] *dvalues[0][0], weights[1][1]*dvalues[0][1],
          weights[1][2]*dvalues[0][2]])
dx2 = sum([weights[2][0] *dvalues[0][0], weights[2][1]*dvalues[0][1],
          weights[2][2]*dvalues[0][2]])
dx3 = sum([weights[3][0] *dvalues[0][0], weights[3][1]*dvalues[0][1],
          weights[3][2]*dvalues[0][2]])

dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)