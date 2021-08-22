import numpy as np

class NeuralNetwork:

    def __init__(self, layers_dict, learning_rate=0.01, num_iterations=10000, verbose=False):
        """
        Initializes the NeuralNetwork class

        Args:
            layers_dict: A dictionary containing the number of nodes in each layer and its activations
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.layers = layers_dict['layers']
        self.activations = layers_dict['activations']

        # Initialize activation functions
        self._initialize_activation_functions()

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_activation_functions(self):
        """
        Initializes the activation functions
        """
        self.activation_functions = {}
        # Sigmoid
        self.activation_functions['sigmoid'] = lambda x: 1 / (1 + np.exp(-x))
        # ReLU
        self.activation_functions['relu'] = lambda x: np.maximum(0, x)
        # TODO: Add more activation functions

        # Validate activations
        for activation in self.activations:
            if activation not in self.activation_functions:
                raise ValueError('Activation function {} not found'.format(activation))
    
    def _initialize_parameters(self):
        """
        Initializes the parameters for the NeuralNetwork class
        """
        self.parameters = {}
        L = len(self.layers)

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.01
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
        L = len(self.layers_dict)

        for l in range(1, L):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A) + b
            activation_function = self.activations[l]
            A = self.activation_functions[activation_function](Z)

        return A