import numpy as np
import h5py
from pathlib import Path

class NeuralNetwork:

    def __init__(self, layers_dict=None, learning_rate=0.01, verbose=False, verbose_iteration=100):
        """
        Initializes the NeuralNetwork class

        Args:
            layers_dict: A dictionary containing the number of nodes in each layer and its activations
            learning_rate: A float value that specifies the learning rate
            verbose: A boolean value that specifies whether to print the cost every 100 iterations
        """
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.verbose_iteration = verbose_iteration
        if layers_dict is not None:
            self.layers = np.array(layers_dict['layers'])
            self.activations = layers_dict['activations']
        else:
            self.layers = []
            self.activations = []

        # Initialize activation functions
        self._initialize_activation_cost_functions()

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_activation_cost_functions(self):
        """
        Initializes the activation functions
        """
        self.activation_functions = {}
        self.backward_activation_functions = {}
        self.cost_functions = {}
        # Sigmoid
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activation_functions['sigmoid'] = sigmoid
        self.backward_activation_functions['sigmoid'] = lambda dA, Z: dA * (sigmoid(Z) * (1 - sigmoid(Z)))
        # ReLU
        self.activation_functions['relu'] = lambda x: np.maximum(0, x)
        self.backward_activation_functions['relu'] = lambda dA, Z: np.where(Z <= 0, 0, dA)
        # Softmax
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
        self.activation_functions['softmax'] = softmax
        self.backward_activation_functions['softmax'] = lambda dA, Z: dA * softmax(Z)

        # Sigmoid cost function
        self.cost_functions['sigmoid'] = lambda Y, A: -np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))) / Y.shape[1]
        # Softmax cost function
        self.cost_functions['softmax'] = lambda Y, A: -np.sum(np.multiply(Y, np.log(A))) / Y.shape[1]

        # Validate activations
        for activation in self.activations:
            if activation not in self.activation_functions:
                raise ValueError('Activation function {} not found'.format(activation))
    
    def _initialize_parameters(self):
        """
        Initializes the parameters for the NeuralNetwork class
        """
        self.grads = {}
        self.parameters = {}
        L = len(self.layers)

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) / np.sqrt(self.layers[l-1])
            self.parameters['b' + str(l)] = np.zeros((self.layers[l], 1))

    def _forward(self, X):
        """
        Performs forward propagation

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
        Returns:
            A numpy.ndarray with shape (nx, m) containing the output of the forward propagation
        """
        A = X
        L = len(self.layers)
        for l in range(1, L):
            self.parameters['A' + str(l-1)] = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A) + b
            self.parameters['Z' + str(l)] = Z
            activation_function = self.activations[l-1]
            A = self.activation_functions[activation_function](Z)

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
        cost = self.cost_functions[self.activations[-1]](Y, A)
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

    def fit(self, X, Y, epochs=1, validation_data=None):
        """
        Trains the model

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        """
        
        for i in range(epochs):
            A = self._forward(X)
            cost = self._cost(Y, A)
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

        # If softmax, get the most likely class
        if self.activations[-1] == 'softmax':
            Y_prediction = np.argmax(Y_prediction, axis=0)
        # If sigmoid, get the sigmoid output
        else:
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
        # Predict labels
        Y_prediction = self.predict(X)

        # If sofmax, get argmax of Y
        if self.activations[-1] == 'softmax':
            _Y = np.argmax(Y, axis=0)
        else:
            _Y = Y
        
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
        filename = filename + '.hdf5'
        f = h5py.File(filename, 'w')
        for l in range(1, len(self.layers)):
            # Save weights
            f.create_dataset('W' + str(l), data=self.parameters['W' + str(l)])
            f.create_dataset('b' + str(l), data=self.parameters['b' + str(l)])
            # Save grads
            f.create_dataset('grads_W' + str(l), data=self.grads['dW' + str(l)])
            f.create_dataset('grads_b' + str(l), data=self.grads['db' + str(l)])
        # Save activation functions
        f.create_dataset('activations', data=self.activations)
        # Save layers
        f.create_dataset('layers', data=self.layers)
        f.close()

    def load(self, filename):
        """
        Loads the model from a file

        Args:
            filename: A string containing the path to the file
        """
        f = h5py.File(f'{filename}.hdf5', 'r')
        self.activations = f['activations'][()]
        self.activations = [activation.decode('utf-8') for activation in self.activations]
        self.layers = f['layers'][()]
        for l in range(1, len(self.layers)):
            self.parameters['W' + str(l)] = f['W' + str(l)][()]
            self.parameters['b' + str(l)] = f['b' + str(l)][()]
            self.grads['dW' + str(l)] = f['grads_W' + str(l)][()]
            self.grads['db' + str(l)] = f['grads_b' + str(l)][()]
        f.close()
