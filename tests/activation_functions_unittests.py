"""
Unit tests for Activation Functions
"""

import unittest
import numpy as np
import logging
import os

from dl_workshop.activation_functions import (
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    leaky_relu_custom,
    softmax,
)
from tests.utils import _log_test_title

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger("Activation Functions unittest")


class ActivationFunctionTests(unittest.TestCase):
    """
    Unit tests for Activation Functions
    """

    def test_sigmoid(self):
        """
        Test sigmoid function
        """
        _log_test_title("Test sigmoid function", logger)
        x = np.array([-100, -1, 0, 1, 100])
        y = sigmoid(x)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.26894142, 0.5, 0.73105858, 1.0]))

        # Test derivative
        x = np.array([-100, -1, 0, 1, 100])
        y = sigmoid(x, deriv=True)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.19661193, 0.25, 0.19661193, 0.0]))

    def test_tanh(self):
        """
        Test tanh function
        """
        _log_test_title("Test tanh function", logger)
        x = np.array([-100, -1, 0, 1, 100])
        y = tanh(x)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [-1.0, -0.76159416, 0.0, 0.76159416, 1.0]))

        # Test derivative
        x = np.array([-100, -1, 0, 1, 100])
        y = tanh(x, deriv=True)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.41997434, 1.0, 0.41997434, 0.0]))

    def test_relu(self):
        """
        Test relu function
        """
        _log_test_title("Test relu function", logger)
        x = np.array([-100, -1, 0, 1, 100])
        y = relu(x)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.0, 0.0, 1.0, 100.0]))

        # Test derivative
        x = np.array([-100, -1, 0, 1, 100])
        y = relu(x, deriv=True)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.0, 0.0, 1.0, 1.0]))

    def test_leaky_relu(self):
        """
        Test leaky relu function
        """
        _log_test_title("Test leaky relu function", logger)
        x = np.array([-100, -1, 0, 1, 100])
        y = leaky_relu(x)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [-1.0, -0.01, 0.0, 1.0, 100.0]))

        # Test derivative
        x = np.array([-100, -1, 0, 1, 100])
        y = leaky_relu(x, deriv=True)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.01, 0.01, 0.01, 1.0, 1.0]))

    def test_leaky_relu_custom(self):
        """
        Test leaky relu function
        """
        _log_test_title("Test leaky relu function", logger)
        x = np.array([-100, -1, 0, 1, 100])
        y = leaky_relu_custom(x)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [-1.0, -0.01, 0.0, 1.0, 100.0]))

        # Test derivative
        x = np.array([-100, -1, 0, 1, 100])
        y = leaky_relu_custom(x, deriv=True)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.0, 0.0, 1.0, 1.0]))

    def test_softmax(self):
        """
        Test softmax function
        """
        _log_test_title("Test softmax function", logger)
        x = np.array([-100, -1, 0, 1, 100])
        y = softmax(x)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.0, 0.0, 0.0, 1.0]))

        # Test derivative
        x = np.array([-100, -1, 0, 1, 100])
        y = softmax(x, deriv=True)
        logger.debug("x = %s", x)
        logger.debug("y = %s", y)
        self.assertTrue(np.allclose(y, [0.0, 0.0, 0.0, 0.0, 1.0]))
