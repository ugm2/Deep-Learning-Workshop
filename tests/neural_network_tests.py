'''
Unit tests for Neural Network
'''

import unittest
import numpy as np
from unittest import mock
import h5py
import numpy as np
import logging
import os
from pathlib import Path

from dl_workshop.neural_network import NeuralNetwork

from tests.utils import _log_test_title

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger("Neural Network tests")

class NeuralNetworkUnitTests(unittest.TestCase):
    '''
    Unit tests for Neural Network
    '''

    @classmethod
    def setUpClass(cls):
        """
        Set up all tests
        """
        super(NeuralNetworkUnitTests, cls).setUpClass()

    def setUp(self):
        """
        Set up all tests
        """
        np.random.seed(42)

    def test_init(self):
        '''
        Test Neural Network initialization
        '''
        _log_test_title("Neural Network initialization", logger)

        input_size = 2
        layers = [(2, lambda x: x), (1, lambda x: x)]
        learning_rate = 0.1
        cost_function = lambda x: x
        verbose = True
        verbose_iteration = 100

        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration
        )
        self.assertEqual(nn.input_size, input_size)
        self.assertEqual(nn.layers, layers)
        self.assertEqual(nn.learning_rate, learning_rate)
        self.assertEqual(nn.cost_function, cost_function)
        self.assertEqual(nn.verbose, verbose)
        self.assertEqual(nn.verbose_iteration, verbose_iteration)

    def test_initialize_parameters(self):
        '''
        Test Neural Network initialization
        '''
        _log_test_title("Neural Network initialization", logger)

        input_size = 2
        layers = [(2, lambda x: x), (1, lambda x: x)]
        learning_rate = 0.1
        cost_function = lambda x: x
        verbose = True
        verbose_iteration = 100

        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration
        )
        nn._initialize_parameters()
        # Check shapes
        self.assertEqual(nn.parameters["W1"].shape, (2, 2))
        self.assertEqual(nn.parameters["b1"].shape, (2, 1))
        self.assertEqual(nn.parameters["W2"].shape, (1, 2))
        self.assertEqual(nn.parameters["b2"].shape, (1, 1))
        # Check values 
        assert np.allclose(nn.parameters["W1"], np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]))
        assert np.allclose(nn.parameters["b1"], np.array([[0.], [0.]]))
        assert np.allclose(nn.parameters["W2"], np.array([[-0.32768579, -0.32932067]]))
        assert np.allclose(nn.parameters["b2"], np.array([[0.]]))
        
        self.assertEqual(nn.grads, {})

    def test_forward(self):
        '''
        Test Neural Network forward propagation
        '''
        _log_test_title("Neural Network forward propagation", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, lambda x: x), (1, lambda x: x)]
        learning_rate = 0.1
        cost_function = lambda x: x
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration
        )
        # Mock initialisation of parameters
        nn.parameters = {
            "W1": np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
            "b1": np.array([[0.], [0.]]),
            "W2": np.array([[-0.32768579, -0.32932067]]),
            "b2": np.array([[0.]])
        }
        nn.grads = {}
        # Create fake input data
        X = np.array([[0.05, 0.1], [0.9, 0.65]])
        Y = np.array([[1.0, 0.0]])
        # Forward propagation
        A2 = nn._forward(X)
        # Check shapes
        self.assertEqual(nn.parameters["A0"].shape, (2, 2))
        self.assertEqual(nn.parameters["Z1"].shape, (2, 2))
        self.assertEqual(nn.parameters["A1"].shape, (2, 2))
        self.assertEqual(nn.parameters["Z2"].shape, (1, 2))
        self.assertEqual(A2.shape, (1, 2))
        # Check values
        assert np.allclose(nn.parameters['A0'], np.array([[0.05, 0.1], [0.9, 0.65]]))
        assert np.allclose(nn.parameters['Z1'], np.array([[0.54422607, 0.4643951], [0.32868467, 0.21617428]]))
        assert np.allclose(nn.parameters['A1'], np.array([[0.54422607, 0.4643951], [0.32868467, 0.21617428]]))
        assert np.allclose(nn.parameters['Z2'], np.array([[-0.28657781, -0.22336633]]))
        assert np.allclose(A2, np.array([[-0.28657781, -0.22336633]]))
        

