import numpy as np
from pathlib import Path
import pickle

class NeuralNetwork:

    def __init__(self,
                 n_inputs,
                 layers,
                 cost_function,
                 learning_rate=0.01,
                 verbose=False,
                 verbose_iteration=100):
        """
        Initializes the NeuralNetwork class

        Args:
            n_inputs: The number of inputs
            layers: A list of tuples of integers that represent the number
                    of nodes in each layer and activation function
            cost_function: The cost function to use
            learning_rate: A float value that specifies the learning rate
            verbose: A boolean value that specifies whether to print the cost
            verbose_iteration: An integer value that specifies the number of iterations to output the cost
        """
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.verbose_iteration = verbose_iteration
        self.layers = layers
        self.n_inputs = n_inputs
        self.cost_function = cost_function

        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initializes the parameters for the NeuralNetwork class
        """
        self.grads = {}
        self.parameters = {}
        layers = [(self.n_inputs, None)] + self.layers

        for l in range(1, len(layers)):
            self.parameters['W' + str(l)] = np.random.randn(layers[l][0],
                                                            layers[l-1][0]) / \
                                                                np.sqrt(layers[l-1][0])
            self.parameters['b' + str(l)] = np.zeros((layers[l][0], 1))

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

    def _backward(self, A, Y):
        """
        Performs backward propagation

        Args:
            A: A numpy.ndarray with shape (1, m) containing the predictions
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        Returns:
            dA: A numpy.ndarray with shape (nx, m) that contains the gradients of the cost with respect to A
        """

        # Initialize gradient dA
        dA = self.cost_function(Y, A, deriv=True)

        L = len(self.layers)

        for l in reversed(range(0, L)):
            A_prev = self.parameters['A' + str(l)]
            W = self.parameters['W' + str(l+1)]
            Z = self.parameters['Z' + str(l+1)]
            dZ = dA * self.layers[l][1](Z, deriv=True)
            dA = np.dot(W.T, dZ)
            self.grads['dW' + str(l+1)] = np.dot(dZ, A_prev.T) / A_prev.shape[1]
            self.grads['db' + str(l+1)] = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]

    def _update_parameters(self):
        """
        Updates the parameters using gradient descent
        """
        L = len(self.layers)
        for l in range(1, L+1):
            self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - self.learning_rate * self.grads['dW' + str(l)]
            self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - self.learning_rate * self.grads['db' + str(l)]

    def fit(self, X, Y, epochs=1, validation_data=None):
        """
        Trains the model

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        """
        
        for i in range(epochs):
            A = self._forward(X)
            cost = self.cost_function(Y, A)
            self._backward(A, Y)
            self._update_parameters()
            if self.verbose and i % self.verbose_iteration == 0:
                print('Iteration {}: Cost = {}'.format(i, cost))

        # Print accuracy if verbose
        if self.verbose:
            print('Train accuracy: {}'.format(self.evaluate(X, Y)['accuracy']))
            if validation_data:
                print('Validation accuracy: {}'.format(self.evaluate(validation_data[0], validation_data[1])['accuracy']))

    def predict(self, X):
        """
        Predicts the output of the model for a set of inputs

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
        Returns:
            Y_prediction: A numpy.ndarray with shape (1, m) that contains the predictions
        """
        Y_prediction = self._forward(X)

        # If binary classification, return 0 or 1
        if Y_prediction.shape[0] == 1:
            return np.where(Y_prediction > 0.5, 1, 0)
        # If multiclass classification, return the index of the highest probability
        else:
            return np.argmax(Y_prediction, axis=0)

    def evaluate(self, X, Y):
        """
        Evaluates the model's predictions

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        Returns:
            accuracy: A float value containing the accuracy of the model's predictions
        """
        # Predict labels
        Y_prediction = self.predict(X)

        # If binary classification
        if len(Y.shape) == 2:
            _Y = Y
        # If multiclass classification
        else:
            _Y = np.argmax(Y, axis=0)
        
        # Obtain accuracy, precision, recall, and F1 score
        accuracy = np.sum(Y_prediction == _Y) / Y.shape[1]
        precision = np.sum(np.logical_and(Y_prediction == _Y, _Y == 1)) / np.sum(_Y == 1)
        recall = np.sum(np.logical_and(Y_prediction == _Y, _Y == 1)) / np.sum(_Y == 1)
        f1 = 2 * precision * recall / (precision + recall)
        
        # Return as a dictionary
        return {'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2)}

    def save(self, filename):
        """
        Saves the model to a file

        Args:
            filename: A string containing the path to the file
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename + '.pkl', 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(filename):
        """
        Loads a model from a file

        Args:
            filename: A string containing the path to the file
        Returns:
            model: A NeuralNetwork instance
        """
        with open(filename + '.pkl', 'rb') as file:
            return pickle.load(file)
