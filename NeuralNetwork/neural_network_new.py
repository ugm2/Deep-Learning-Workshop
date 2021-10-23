import numpy as np
from copy import deepcopy

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


class NeuralNetwork:

    def __init__(self, input_n, layers, learning_rate=0.01, num_iterations=10000, verbose=False):
        """
        Initializes the NeuralNetwork class

        Args:
            layers_dict: A dictionary containing the number of nodes in each layer and its activations
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.layers = layers
        self.input_n = input_n

        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initializes the parameters for the NeuralNetwork class
        """
        self.grads = {}
        self.parameters = {}
        layers = [(self.input_n, None)] + self.layers

        for i in range(1, len(layers)):
            self.parameters['W' + str(i)] = np.random.randn(layers[i][0], layers[i-1][0]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((layers[i][0], 1))

    def _forward(self, X):
        """
        Performs forward propagation

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
        Returns:
            A numpy.ndarray with shape (nx, m) containing the output of the forward propagation
        """
        A = X
        for i, layer in enumerate(self.layers):
            self.parameters['A' + str(i)] = A
            W = self.parameters['W' + str(i+1)]
            b = self.parameters['b' + str(i+1)]
            Z = np.dot(W, A) + b
            self.parameters['Z' + str(i+1)] = Z
            A = layer[1](Z)

        return A

    def _cost(self, Y, A):
        """
        Computes the cost

        Args:
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
            A: A numpy.ndarray with shape (1, m) that contains the output of the forward propagation
        Returns:
            cost: A float value containing the cost
        """
        # TODO: Generalize and add more cost functions
        cost = - (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T)) / Y.shape[1]
        cost = np.squeeze(cost) 
        return cost

    def _backward(self, A, Y):
        """
        Performs backward propagation

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        Returns:
            dA: A numpy.ndarray with shape (nx, m) that contains the gradients of the cost with respect to A
        """
        # TODO: Generalize for more cost functions and its derivatives

        # Initialize gradient dA
        dA = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))

        L = len(self.layers)

        for l in reversed(range(1, L)):
            A_prev = self.parameters['A' + str(l-1)]
            W = self.parameters['W' + str(l)]
            Z = self.parameters['Z' + str(l)]
            activation_function = self.activations[l-1]
            backward_activation_function = self.backward_activation_functions[activation_function]
            dZ = backward_activation_function(dA, Z)
            dA = np.dot(W.T, dZ)
            self.grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / A_prev.shape[1]
            self.grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]

    def _update_parameters(self):
        """
        Updates the parameters using gradient descent
        """
        L = len(self.layers)
        for l in range(1, L):
            self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - self.learning_rate * self.grads['dW' + str(l)]
            self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - self.learning_rate * self.grads['db' + str(l)]

    def fit(self, X, Y):
        """
        Trains the model

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        """
        
        for i in range(self.num_iterations):
            A = self._forward(X)
            cost = self._cost(Y, A)
            self._backward(A, Y)
            self._update_parameters()
            if self.verbose and i % 100 == 0:
                print('Iteration {}: Cost = {}'.format(i, cost))


        # Print accuracy if verbose
        if self.verbose:
            print('Train accuracy: {}'.format(self.evaluate(X, Y)))

    def predict(self, X):
        """
        Predicts the output of the model for a set of inputs

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
        Returns:
            Y_prediction: A numpy.ndarray with shape (1, m) that contains the predictions
        """

        Y_prediction = self._forward(X)
        Y_prediction = np.where(Y_prediction > 0.5, 1, 0)
        return Y_prediction

    def evaluate(self, X, Y):
        """
        Evaluates the model's predictions

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        Returns:
            accuracy: A float value containing the accuracy of the model's predictions
        """
        Y_prediction = self.predict(X)
        Y_prediction = np.where(Y_prediction > 0.5, 1, 0)
        return np.sum(Y_prediction == Y) / Y.shape[1]

    # TODO: Save, Load

