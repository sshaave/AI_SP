import copy
import numpy as np
import nnfs
import pickle
import matplotlib.pyplot as plt
import time
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
from skopt import BayesSearchCV, gp_minimize
from skopt.plots import plot_convergence, plot_objective_2D
from skopt.space import Real, Categorical, Integer
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skopt.utils import use_named_args

nnfs.init()

# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containg trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the first layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases, will be redundant
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and
        # loss functions is Categorical Cross-Entropy
        # create and object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss function
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            if batch_size is not None:
                print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_param()

                # Print a summary
                if (not step % print_every) and (step == train_steps - 1 and batch_size is not None):
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f}, (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if batch_size is None:
                if not epoch % print_every:
                    print(f'epoch: {epoch}')
                    print(f'training, ' +
                          f'acc: {epoch_accuracy:.3f}, ' +
                          f'loss: {epoch_loss:.3f} (' +
                          f'data_loss: {epoch_data_loss:.3f}, ' +
                          f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')
            if batch_size is not None:
                print(f'training, ' +
                      f'acc: {epoch_accuracy:.3f}, ' +
                      f'loss: {epoch_loss:.3f} (' +
                      f'data_loss: {epoch_data_loss:.3f}, ' +
                      f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

            # If there is validation data
            if validation_data is not None:

                # Evaluate the model
                self.evaluate(*validation_data, epoch=epoch, print_every=print_every, batch_size=batch_size)

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output propert  that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list
        # return its output
        return layer.output

    # Performs backwards pass
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not use backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs propert that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Evaluates the model
    def evaluate(self, X_val, y_val, *, epoch=1, print_every=1, batch_size=None):

        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this wont include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()
        # Iterate over steps
        for step in range(validation_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        # Print a summary
        if batch_size is None:
            if not epoch % print_every:
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')
        else:
            print(f'validation, ' +
                  f'acc: {validation_accuracy:.3f}, ' +
                  f'loss: {validation_loss:.3f}')

        return validation_accuracy

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):

        # Create a list for parameters
        parameters = []

        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a list
        return parameters

    # Updates the model with new parameters
    def set_parameters(self, parameters):

        # Iterate over the parameters and layers
        # and update each layer with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save_parameters(self, path):

        # Open a file in the binary-write mode
        # and save parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):

        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):

        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # Loads and returns a model
    @staticmethod
    def load(path):

        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Return a model
        return model

    # Predicts on the samples
    def predict(self, X, *, batch_size=None):

        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model output
        output = []
        for step in range(prediction_steps):

            if batch_size is None:
                batch_X = X

            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)
# ----------------------------------

# Input "layer"
class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

# Dense layer
class Layer_Dense:

    # Layer definition
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0., weight_regularizer_l2=0.,
                 bias_regularizer_l1=0., bias_regularizer_l2=0.):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2*self.weight_regularizer_l2 * \
                            self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2*self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout
class Layer_Dropout:

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save selected scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Common accuracy class
class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison values
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):

    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculate precision value
    # based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

# Accuracy calculation for classification model
class Accuracy_Catergorial(Accuracy):

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    # Compare predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return(outputs)

# Softmax activation
class Activation_Softmax:

    def forward(self, inputs, training):

        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid activation
class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs, training):
        # Save input and calculate/save outout
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# Linear activation
class Activation_Linear:

    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# SGD optimizer: general starting LR = 1.0, with a decay down to 0.1
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # Check if momentum is used
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # if there is no momentum array for weights
                # the array doesn't exist for biases yet either
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor
            # and update with current gradient
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

            # Vanilla SGD updates (as before momentum updates)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases +=  bias_updates

    # Call once after any parameter update
    def post_update_param(self):
        self.iterations += 1

# Adagrad optimizer:
class Optimizer_Adagrad:

    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # if there is no cache array for weights
            # the array doesn't exist for biases yet either
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)


    # Call once after any parameter update
    def post_update_param(self):
        self.iterations += 1

# RMSprop optimizer:
class Optimizer_RMSprop:

    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # if there is no cache array for weights
            # the array doesn't exist for biases yet either
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                              (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                        layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)


    # Call once after any parameter update
    def post_update_param(self):
        self.iterations += 1

# Adam optimizer: good starting LR = 1e-3, decaying down to 1e-4
class Optimizer_Adam:

    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            # if there is no cache array for weights
            # the array doesn't exist for biases yet either
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentums with current gradients
        layer.weight_momentums = self.beta_1 * \
                                layer.weight_momentums + \
                                (1 - self.beta_1) * layer.dweights

        layer.bias_momentums = self.beta_1 * \
                                layer.bias_momentums + \
                                (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                        weight_momentums_corrected / \
                        (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)


    # Call once after any parameter update
    def post_update_param(self):
        self.iterations += 1

# Common loss class
class Loss:

    # Regularization loss calculation
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                        np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return loss and regularization loss
        return data_loss, self.regularization_loss()

    # Calculate accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probability for target values -
        # only if categorical labels
        if len(y_true.shape) == 1: # /1/
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # If not any of these two
        else:
            print("If statement /1/ not met")
            exit()
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate the gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossEntropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error loss
class Loss_MeanSquaredError(Loss):   # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We''ll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize the gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We''ll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
# --------------------------------------------
# FLAGS
TRAIN_MODEL = False
LOAD_MODEL = False
MINIMAL_EXAMPLE = False
EXAMPLE_10 = False
EXAMPLE_11 = True
DO_BAY_OPT = False
# --------------------------------------------
if TRAIN_MODEL:
    X, y = spiral_data(samples=100, classes=3)
    X_test, y_test = spiral_data(samples=100, classes=3)

    # Reshape labels to be a list of lists
    # Inner list contains one output (either 0 or 1)
    # per each output neuron, 1 in this case
    #y = y.reshape(-1, 1)
    #y_test = y_test.reshape(-1, 1)

    # Instantiate the model
    model = Model()

    # Add layers
    model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.1))
    model.add(Layer_Dense(512, 3))
    model.add(Activation_Softmax())

    # Set loss and optimizer objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
        accuracy=Accuracy_Catergorial()
    )

    # Finalize the model
    model.finalize()

    # Train the model
    model.train(X, y, validation_data=(X_test, y_test), epochs=1000, print_every=100)

    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Get parameters from model to save
    parameters = model.get_parameters()

    # Save models parameters (weights and biases)
    model.save_parameters('notMNIST.parms')

    # load model
    model.load_parameters('notMNIST.parms')

    # save the whole model
    model.save('fullModelnotMNIST.model')

# load the whole model
if LOAD_MODEL:
    X_train, y_train = spiral_data(samples=100, classes=3)
    model = Model.load('fullModelnotMNIST.model')
    confidences = model.predict(X_train[:5])
    predictions = model.output_layer_activation.predictions(confidences)
    print(predictions)

    spiral_labels = {0: 'blue', 1: 'red', 2: 'green'}

    for prediction in predictions:
        print(spiral_labels[prediction])

# Minimal example
if MINIMAL_EXAMPLE:
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(
        SVC(),
        {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'gamma': (1e-6, 1e+1, 'log-uniform'),
            'degree': (1, 8),  # integer valued parameter
            'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
        },
        n_iter=32,
        cv=3
    )

    opt.fit(X_train, y_train)

# Example 10
if EXAMPLE_10:
    print('Example 10')
    search_space = {'wrl1': Integer(5e-5, 5e-4, 'log-uniform'),
                    'brl1': Integer(5e-5, 5e-4, 'log-uniform'),
                    'wrl2': Integer(5e-5, 5e-4, 'log-uniform'),
                    'brl2': Integer(5e-5, 5e-4, 'log-uniform')
                  }
    def on_step(optim_result):
        score = forest_bayes_search.best_score_
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True


    forest_bayes_search = BayesSearchCV(model, search_space, n_iter=32,
                                        scoring="accuracy", n_jobs=-1, cv=5)
    forest_bayes_search.fit(X_train, y_train, callback=on_step)

if EXAMPLE_11:
    print('Example 11')

    dim_wr1 = Real(low=1e-8, high=1e-5, prior='log-uniform', name='wr1')
    dim_br1 = Real(low=1e-8, high=1e-5, prior='log-uniform', name='br1')
    dim_wr2 = Real(low=1e-8, high=1e-5, prior='log-uniform', name='wr2')
    dim_br2 = Real(low=1e-8, high=1e-5, prior='log-uniform', name='br2')
    dimensions = [dim_wr1, dim_br1, dim_wr2, dim_br2]

    def create_model(wr1, br1, wr2, br2):
        model = Model()
        # Add layers
        model.add(Layer_Dense(2, 64, weight_regularizer_l1=wr1, bias_regularizer_l1=br1, weight_regularizer_l2=wr2, bias_regularizer_l2=br2))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(64, 124))
        model.add(Activation_Softmax())

        # Set loss and optimizer objects
        model.set(
            loss=Loss_CategoricalCrossentropy(),
            optimizer=Optimizer_Adam(learning_rate=0.028, decay=5e-6),
            accuracy=Accuracy_Catergorial()
        )

        # Finalize the model
        model.finalize()

        return model

    @use_named_args(dimensions=dimensions)
    def fitness(wr1, br1, wr2, br2):
        global X_training, y_training, X_val, y_val
        # Print the hyper-parameters
        print('weight regularizer L1: {0:.2e}'.format(wr1))
        print('bias regularizer L1: {0:.2e}'.format(br1))
        print('weight regularizer L2: {0:.2e}'.format(wr2))
        print('bias regularizer L2: {0:.2e}'.format(br2))

        # Create the neural network
        model = create_model(wr1, br1, wr2, br2)
        model.train(X_training, y_training, validation_data=(X_val, y_val), epochs=1000, print_every=100)

        return model.evaluate(X_val, y_val)

    def readTrainingData():
        with open("SP/trainingData_pickle.pk", 'rb') as fi1:
            X_training = pickle.load(fi1)
        with open("SP/trainingData_Y_pickle.pk", 'rb') as fi2:
            y_training = pickle.load(fi2)
        return X_training, y_training

    def readTestData():
        with open("SP/testData_pickle.pk", 'rb') as fti1:
            X_test = pickle.load(fti1)
        with open("SP/testData_Y_pickle.pk", 'rb') as fti2:
            y_test = pickle.load(fti2)
        return X_test, y_test

    # Default parameters
    X_training, y_training = readTrainingData()
    X_val, y_val = readTestData()
    default_parameters = [1e-5, 1e-5, 2e-8, 2e-8]


    #fitness(x=default_parameters)
    # default yields accuracy = 0.710, loss 0.820
    if DO_BAY_OPT:
        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func="EI",
                                    n_calls=18,
                                    x0=default_parameters)


        with open("search_result_pickle.pk", 'wb') as f:
            pickle.dump(search_result, f)
    else:
        with open("search_result_pickle.pk", "rb") as f:
            search_result = pickle.load(f)


    #plot_convergence(search_result)
    #plt.savefig("Convergence.png", dpi=400)
    print('Search_result.x:')
    print(search_result.x)

    print("sorted(zip(search_result.func_vals, search_result.x_iters))")
    print(sorted(zip(search_result.func_vals, search_result.x_iters)))

    fig = plot_objective_2D(result=search_result,
                            dimension_identifier1='wr1',
                            dimension_identifier2='wr2',
                            levels=50)
    plt.savefig("wrL.png", dpi=400)
    plt.show()
    '''
    Search_result.x:
    [1.22724151191499e-08, 1.3244871554343121e-07, 5.071166851500468e-08, 1.910295036087759e-06]
    sorted(zip(search_result.func_vals, search_result.x_iters))
    [(0.7161, [1.22724151191499e-08, 1.3244871554343121e-07, 5.071166851500468e-08, 1.910295036087759e-06]), (0.7493, [8.305853730253613e-08, 7.001500192875131e-07, 1.0537205016378386e-08, 1.4029743777425741e-08]), (0.7859, [9.913296673308842e-06, 1.6882470948611348e-08, 9.328715812364817e-06, 1.497869953434122e-06]), (0.7931, [1.3322394097805917e-08, 9.739960940438176e-07, 2.284192411319548e-08, 3.883678696071521e-08]), (0.7952, [1.1972066989790358e-07, 1.7256983760104936e-06, 1.0030485792035732e-08, 3.9297435379102864e-08]), (0.7971, [1.1531615597627458e-08, 7.382859645867013e-07, 1.917388681361202e-07, 1.2034603632343203e-08]), (0.8027, [1.2309948575041805e-08, 5.966299097426005e-08, 3.4534872266360856e-08, 1.9385064527091324e-06]), (0.804, [5.582393385102372e-07, 1.4394296220367527e-08, 3.971710356911856e-08, 5.2348861644816967e-08]), (0.8041, [1.1198104737350428e-08, 6.955024868027683e-07, 1e-08, 2.291573590896186e-08]), (0.8042, [1.4452810458573e-07, 6.470948482960243e-07, 2.723812009178251e-07, 4.472603616433414e-06]), (0.8043, [6.544038108389113e-06, 9.056335546321557e-06, 9.424065355747783e-06, 1.66412939607391e-07]), (0.8062, [1.4491060717626565e-07, 6.87339719123175e-06, 1.1240482900879422e-08, 4.275091387883615e-08]), (0.8088, [9.754755907789272e-08, 3.5057427435110065e-07, 1.0638387215977429e-08, 7.8248905368655e-06]), (0.8101, [1.8013195327882917e-07, 2.584536081026979e-07, 1.0138518023883385e-08, 3.878628409330823e-08]), (0.8133, [9.28762163250718e-08, 9.621973271775866e-08, 4.700464389312515e-08, 6.132547246328457e-06]), (0.8178, [1.3925144991828371e-08, 8.327397881225184e-07, 1.7701862488859076e-07, 5.500640182247603e-06]), (0.8193, [1.1079232642086768e-08, 1.3005573153479623e-06, 5.751913114338393e-07, 5.651805296848443e-06]), (0.8259, [3.9648859259776015e-07, 2.7474296694207597e-07, 2.1766273774530188e-06, 1.961195568030566e-07]), (0.8274, [1.0147153024900139e-08, 6.599829780487763e-06, 3.737706988083254e-08, 2.708551486101469e-07]), (0.8292, [1.0041023786967327e-08, 1.016753641530317e-08, 1.0129805622854212e-07, 4.3925439911505126e-07]), (0.833, [2.213969303153943e-07, 6.218188390579742e-06, 7.674525507928192e-06, 2.2107051766268668e-07]), (0.8344, [2.332208497925173e-06, 2.699495046309555e-06, 1.0914062396405989e-08, 6.938372183192817e-06]), (0.8346, [3.7277567344967455e-07, 8.631756613123617e-06, 1.4675033479595774e-08, 3.3781801220074066e-08]), (0.8412, [4.901003215498935e-06, 2.4662080907135214e-08, 1.636630728975208e-08, 1.302564077388259e-07]), (0.8418, [1.7758809054727003e-08, 2.844148595199579e-07, 2.8645707818593822e-08, 2.6432314004971706e-06]), (0.8483, [4.116091447975427e-06, 3.2106264753129937e-06, 8.329697090553463e-07, 2.6839831888412833e-06]), (0.8494, [2.1902093620039816e-06, 1.0831960696017724e-08, 9.23497813176753e-06, 4.156191273822792e-08]), (0.8505, [4.431902706146644e-08, 6.553680320749538e-06, 1.7806154857131988e-07, 2.702470921628961e-06]), (0.8567, [7.665938237093883e-08, 2.3651992338831916e-08, 1.7220390916804237e-06, 3.6213980263649355e-07]), (0.857, [6.965998096787879e-08, 2.1002800969903118e-07, 1.0440826860175083e-08, 7.407314169925384e-07]), (0.8601, [1.1371154067540244e-08, 7.580879655698566e-06, 2.5308168204996135e-06, 5.446876964044745e-08]), (0.8649, [8.383327100038674e-06, 1e-05, 1.196881382937505e-07, 4.451902630672063e-08]), (0.865, [1.3359151148418345e-08, 6.155958007059439e-07, 1e-05, 1e-08]), (0.8673, [3.496780187976165e-08, 2.614643638666961e-07, 1.2349013163271181e-06, 1.5283887536059276e-07]), (0.8723, [8.170347858570898e-07, 1.1472249991858674e-06, 3.942081078728906e-06, 3.317548704066968e-07]), (0.8737, [2.3811368885249881e-07, 3.225550247984213e-07, 1.4223611718747952e-07, 4.10721628615726e-08]), (0.8739, [1e-05, 1e-05, 2e-08, 2e-08]), (0.8762, [4.0135643598252096e-08, 4.409967026691704e-07, 2.3623323090811527e-08, 4.7013238387116035e-08]), (0.8828, [1.0504863275938163e-06, 3.338916808942348e-08, 4.315226581463595e-08, 9.338033481541627e-07]), (0.8921, [1.277620484849237e-07, 2.0791615297116458e-08, 9.967492804791076e-06, 1.2929427142262608e-07])]

    '''