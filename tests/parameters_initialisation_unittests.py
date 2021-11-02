"""Unit tests for Parameters Initialisation methods."""

import unittest
import numpy as np
import logging
import os

from dl_workshop.parameters_initialisation import (
    zero_initialisation,
    random_initialisation,
    xavier_initialisation,
    he_initialisation,
)

from tests.utils import _log_test_title

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger("Parameters Initialisation unittest")


class ParametersInitialisationTests(unittest.TestCase):
    """Unit tests for Parameters Initialisation methods."""

    @classmethod
    def setUpClass(cls):
        """Set up all tests."""
        super(ParametersInitialisationTests, cls).setUpClass()

    def setUp(self):
        """Set up all tests."""
        np.random.seed(42)

    def test_zero_initialization(self):
        """Test zero initialization."""
        _log_test_title("Test zero initialization", logger)
        dimensions = [2, 3, 4]
        init_params = zero_initialisation(dimensions)
        # Check init_params keys
        self.assertTrue("parameters" in init_params)
        self.assertTrue("grads" in init_params)
        parameters = init_params["parameters"]
        # Check parameters keys
        self.assertTrue("W1" in parameters)
        self.assertTrue("b1" in parameters)
        self.assertTrue("W2" in parameters)
        self.assertTrue("b2" in parameters)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        assert np.array_equal(W1, np.zeros((dimensions[1], dimensions[0])))
        assert np.array_equal(W2, np.zeros((dimensions[2], dimensions[1])))
        assert np.array_equal(b1, np.zeros((dimensions[1], 1)))
        assert np.array_equal(b2, np.zeros((dimensions[2], 1)))
        grads = init_params["grads"]
        self.assertEqual(grads, {})

    def test_random_initialization(self):
        """Test random initialization."""
        _log_test_title("Test random initialization", logger)
        dimensions = [2, 3, 4]
        init_params = random_initialisation(dimensions)
        # Check init_params keys
        self.assertTrue("parameters" in init_params)
        self.assertTrue("grads" in init_params)
        parameters = init_params["parameters"]
        # Check parameters keys
        self.assertTrue("W1" in parameters)
        self.assertTrue("b1" in parameters)
        self.assertTrue("W2" in parameters)
        self.assertTrue("b2" in parameters)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        np.random.seed(42)
        assert np.array_equal(W1, np.random.randn(dimensions[1], dimensions[0]) * 0.01)
        assert np.array_equal(W2, np.random.randn(dimensions[2], dimensions[1]) * 0.01)
        assert np.array_equal(b1, np.zeros((dimensions[1], 1)))
        assert np.array_equal(b2, np.zeros((dimensions[2], 1)))
        grads = init_params["grads"]
        self.assertEqual(grads, {})

    def test_xavier_initialization(self):
        """Test xavier initialization."""
        _log_test_title("Test xavier initialization", logger)
        dimensions = [2, 3, 4]

        # NORMAL DISTRIBUTION
        init_params = xavier_initialisation(dimensions, distribution="normal")
        # Check init_params keys
        self.assertTrue("parameters" in init_params)
        self.assertTrue("grads" in init_params)
        parameters = init_params["parameters"]
        # Check parameters keys
        self.assertTrue("W1" in parameters)
        self.assertTrue("b1" in parameters)
        self.assertTrue("W2" in parameters)
        self.assertTrue("b2" in parameters)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        np.random.seed(42)
        limit = np.sqrt(2 / (dimensions[0] + dimensions[1]))
        assert np.array_equal(
            W1, np.random.normal(0.0, limit, size=(dimensions[1], dimensions[0]))
        )
        limit = np.sqrt(2 / (dimensions[1] + dimensions[2]))
        assert np.array_equal(
            W2, np.random.normal(0.0, limit, size=(dimensions[2], dimensions[1]))
        )
        assert np.array_equal(b1, np.zeros((dimensions[1], 1)))
        assert np.array_equal(b2, np.zeros((dimensions[2], 1)))
        grads = init_params["grads"]
        self.assertEqual(grads, {})

        # UNIFORM DISTRIBUTION
        np.random.seed(42)
        init_params = xavier_initialisation(dimensions, distribution="uniform")
        # Check init_params keys
        self.assertTrue("parameters" in init_params)
        self.assertTrue("grads" in init_params)
        parameters = init_params["parameters"]
        # Check parameters keys
        self.assertTrue("W1" in parameters)
        self.assertTrue("b1" in parameters)
        self.assertTrue("W2" in parameters)
        self.assertTrue("b2" in parameters)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        np.random.seed(42)
        limit = np.sqrt(6 / (dimensions[0] + dimensions[1]))
        assert np.array_equal(
            W1,
            np.random.uniform(
                low=-limit, high=limit, size=(dimensions[1], dimensions[0])
            ),
        )
        limit = np.sqrt(6 / (dimensions[1] + dimensions[2]))
        assert np.array_equal(
            W2,
            np.random.uniform(
                low=-limit, high=limit, size=(dimensions[2], dimensions[1])
            ),
        )
        assert np.array_equal(b1, np.zeros((dimensions[1], 1)))
        assert np.array_equal(b2, np.zeros((dimensions[2], 1)))
        grads = init_params["grads"]
        self.assertEqual(grads, {})

        # DIFFERENT DISTRIBUTION
        # Check it fails
        with self.assertRaises(ValueError):
            xavier_initialisation(dimensions, distribution="invalid")

    def test_he_initialization(self):
        """Test he initialization."""
        _log_test_title("Test he initialization", logger)
        dimensions = [2, 3, 4]

        # NORMAL DISTRIBUTION
        init_params = he_initialisation(dimensions, distribution="normal")
        # Check init_params keys
        self.assertTrue("parameters" in init_params)
        self.assertTrue("grads" in init_params)
        parameters = init_params["parameters"]
        # Check parameters keys
        self.assertTrue("W1" in parameters)
        self.assertTrue("b1" in parameters)
        self.assertTrue("W2" in parameters)
        self.assertTrue("b2" in parameters)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        np.random.seed(42)
        limit = np.sqrt(2 / dimensions[0])
        assert np.array_equal(
            W1, np.random.normal(0.0, limit, size=(dimensions[1], dimensions[0]))
        )
        limit = np.sqrt(2 / dimensions[1])
        assert np.array_equal(
            W2, np.random.normal(0.0, limit, size=(dimensions[2], dimensions[1]))
        )
        assert np.array_equal(b1, np.zeros((dimensions[1], 1)))
        assert np.array_equal(b2, np.zeros((dimensions[2], 1)))
        grads = init_params["grads"]
        self.assertEqual(grads, {})

        # UNIFORM DISTRIBUTION
        np.random.seed(42)
        init_params = he_initialisation(dimensions, distribution="uniform")
        # Check init_params keys
        self.assertTrue("parameters" in init_params)
        self.assertTrue("grads" in init_params)
        parameters = init_params["parameters"]
        # Check parameters keys
        self.assertTrue("W1" in parameters)
        self.assertTrue("b1" in parameters)
        self.assertTrue("W2" in parameters)
        self.assertTrue("b2" in parameters)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        np.random.seed(42)
        limit = np.sqrt(6 / dimensions[0])
        assert np.array_equal(
            W1,
            np.random.uniform(
                low=-limit, high=limit, size=(dimensions[1], dimensions[0])
            ),
        )
        limit = np.sqrt(6 / dimensions[1])
        assert np.array_equal(
            W2,
            np.random.uniform(
                low=-limit, high=limit, size=(dimensions[2], dimensions[1])
            ),
        )
        assert np.array_equal(b1, np.zeros((dimensions[1], 1)))
        assert np.array_equal(b2, np.zeros((dimensions[2], 1)))
        grads = init_params["grads"]
        self.assertEqual(grads, {})

        # DIFFERENT DISTRIBUTION
        # Check it fails
        with self.assertRaises(ValueError):
            he_initialisation(dimensions, distribution="invalid")
