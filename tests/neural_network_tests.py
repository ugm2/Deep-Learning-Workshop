"""
Unit tests for Neural Network
"""

import pickle
import unittest
import numpy as np
from unittest import mock
import h5py
import numpy as np
import logging
import os
from pathlib import Path

from dl_workshop.neural_network import NeuralNetwork
from dl_workshop.activation_functions import sigmoid, relu, softmax
from dl_workshop.cost_functions import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets.mnist import load_data as mnist_load_data

from tests.utils import _log_test_title

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger("Neural Network tests")


class NeuralNetworkUnitTests(unittest.TestCase):
    """
    Unit tests for Neural Network
    """

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
        """
        Test Neural Network initialization
        """
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
            verbose_iteration=verbose_iteration,
        )
        self.assertEqual(nn.input_size, input_size)
        self.assertEqual(nn.layers, layers)
        self.assertEqual(nn.learning_rate, learning_rate)
        self.assertEqual(nn.cost_function, cost_function)
        self.assertEqual(nn.verbose, verbose)
        self.assertEqual(nn.verbose_iteration, verbose_iteration)

    def test_initialize_parameters(self):
        """
        Test Neural Network initialization
        """
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
            verbose_iteration=verbose_iteration,
        )
        nn._initialize_parameters()
        # Check shapes
        self.assertEqual(nn.parameters["W1"].shape, (2, 2))
        self.assertEqual(nn.parameters["b1"].shape, (2, 1))
        self.assertEqual(nn.parameters["W2"].shape, (1, 2))
        self.assertEqual(nn.parameters["b2"].shape, (1, 1))
        # Check values
        assert np.allclose(
            nn.parameters["W1"],
            np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
        )
        assert np.allclose(nn.parameters["b1"], np.array([[0.0], [0.0]]))
        assert np.allclose(nn.parameters["W2"], np.array([[-0.32768579, -0.32932067]]))
        assert np.allclose(nn.parameters["b2"], np.array([[0.0]]))

        self.assertEqual(nn.grads, {})

    def test_forward(self):
        """
        Test Neural Network forward propagation
        """
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
            verbose_iteration=verbose_iteration,
        )
        # Mock initialisation of parameters
        nn.parameters = {
            "W1": np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
            "b1": np.array([[0.0], [0.0]]),
            "W2": np.array([[-0.32768579, -0.32932067]]),
            "b2": np.array([[0.0]]),
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
        assert np.allclose(nn.parameters["A0"], np.array([[0.05, 0.1], [0.9, 0.65]]))
        assert np.allclose(
            nn.parameters["Z1"],
            np.array([[0.54422607, 0.4643951], [0.32868467, 0.21617428]]),
        )
        assert np.allclose(
            nn.parameters["A1"],
            np.array([[0.54422607, 0.4643951], [0.32868467, 0.21617428]]),
        )
        assert np.allclose(nn.parameters["Z2"], np.array([[-0.28657781, -0.22336633]]))
        assert np.allclose(A2, np.array([[-0.28657781, -0.22336633]]))

    def test_backward(self):
        """
        Test Neural Network backward propagation
        """
        _log_test_title("Neural Network backward propagation", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, lambda x, deriv: x), (1, lambda x, deriv: x)]
        learning_rate = 0.1
        cost_function = lambda Y, A, deriv: Y - A
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration,
        )
        # Mock initialisation of parameters
        nn.parameters = {
            "W1": np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
            "b1": np.array([[0.0], [0.0]]),
            "W2": np.array([[-0.32768579, -0.32932067]]),
            "b2": np.array([[0.0]]),
        }
        nn.grads = {}
        # Create fake input data
        X = np.array([[0.05, 0.1], [0.9, 0.65]])
        Y = np.array([[1.0, 0.0]])
        # Mock forward propagation
        nn.parameters["A0"] = X
        nn.parameters["Z1"] = np.array(
            [[0.54422607, 0.4643951], [0.32868467, 0.21617428]]
        )
        nn.parameters["A1"] = np.array(
            [[0.54422607, 0.4643951], [0.32868467, 0.21617428]]
        )
        nn.parameters["Z2"] = np.array([[-0.28657781, -0.22336633]])
        A2 = np.array([[-0.28657781, -0.22336633]])
        # Backward propagation
        nn._backward(A2, Y)
        # Check shapes
        self.assertEqual(nn.grads["dW1"].shape, (2, 2))
        self.assertEqual(nn.grads["db1"].shape, (2, 1))
        self.assertEqual(nn.grads["dW2"].shape, (1, 2))
        self.assertEqual(nn.grads["db2"].shape, (1, 1))
        # Check values
        assert np.allclose(
            nn.grads["dW1"],
            np.array([[0.00202345, 0.03205639], [0.00117533, 0.01911367]]),
        )
        assert np.allclose(nn.grads["db1"], np.array([[0.03667271], [0.02173073]]))
        assert np.allclose(nn.grads["dW2"], np.array([[-0.11191426, -0.06598652]]))
        assert np.allclose(nn.grads["db2"], np.array([[-0.20929858]]))

    def test_update_parameters(self):
        """
        Test Neural Network update parameters
        """
        _log_test_title("Neural Network update parameters", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, lambda x, deriv: x), (1, lambda x, deriv: x)]
        learning_rate = 0.1
        cost_function = lambda Y, A, deriv: Y - A
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration,
        )
        # Mock initialisation of parameters
        nn.parameters = {
            "W1": np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
            "b1": np.array([[0.0], [0.0]]),
            "W2": np.array([[-0.32768579, -0.32932067]]),
            "b2": np.array([[0.0]]),
        }
        nn.grads = {
            "dW1": np.array([[0.00202345, 0.03205639], [0.00117533, 0.01911367]]),
            "db1": np.array([[0.03667271], [0.02173073]]),
            "dW2": np.array([[-0.11191426, -0.06598652]]),
            "db2": np.array([[-0.20929858]]),
        }
        # Update parameters
        nn._update_parameters()
        # Check shapes
        self.assertEqual(nn.parameters["W1"].shape, (2, 2))
        self.assertEqual(nn.parameters["b1"].shape, (2, 1))
        self.assertEqual(nn.parameters["W2"].shape, (1, 2))
        self.assertEqual(nn.parameters["b2"].shape, (1, 1))
        # Check values
        assert np.allclose(
            nn.parameters["W1"],
            np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]])
            - learning_rate * nn.grads["dW1"],
        )
        assert np.allclose(
            nn.parameters["b1"],
            np.array([[0.0], [0.0]]) - learning_rate * nn.grads["db1"],
        )
        assert np.allclose(
            nn.parameters["W2"],
            np.array([[-0.32768579, -0.32932067]]) - learning_rate * nn.grads["dW2"],
        )
        assert np.allclose(
            nn.parameters["b2"], np.array([[0.0]]) - learning_rate * nn.grads["db2"]
        )

    @mock.patch("dl_workshop.neural_network.NeuralNetwork._forward")
    def test_predict(self, mock_forward):
        """
        Test Neural Network predict
        """
        _log_test_title("Neural Network predict", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, lambda x, deriv: x), (1, lambda x, deriv: x)]
        learning_rate = 0.1
        cost_function = lambda Y, A, deriv: Y - A
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration,
        )
        # Make fake data
        X = np.array([[0.05, 0.1], [0.9, 0.65]])
        Y = np.array([[0.0, 1.0]])
        # Mock forward propagation
        mock_forward.return_value = np.array([[0.5, 0.8]])
        # Predict
        y_pred = nn.predict(X)
        # Check shapes
        self.assertEqual(y_pred.shape, (1, 2))
        # Check values
        assert np.allclose(y_pred, np.array([[0, 1]]))

    @mock.patch("dl_workshop.neural_network.NeuralNetwork.predict")
    def test_evaluate(self, mock_predict):
        """
        Test Neural Network evaluate
        """
        _log_test_title("Neural Network evaluate", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, lambda x, deriv: x), (1, lambda x, deriv: x)]
        learning_rate = 0.1
        cost_function = lambda Y, A, deriv: Y - A
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration,
        )
        # Make fake data
        X = np.array([[0.05, 0.1, 0.0], [0.9, 0.65, 0.0]])
        Y = np.array([[0.0, 1.0, 1.0]])
        # Mock predict
        mock_predict.return_value = np.array([[0, 1, 0]])
        # Evaluate
        metrics = nn.evaluate(X, Y)
        # Check metric values
        self.assertEqual(metrics["accuracy"], 0.67)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.67)

    def test_save(self):
        """
        Test Neural Network save
        """
        _log_test_title("Neural Network save", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, sigmoid), (1, sigmoid)]
        learning_rate = 0.1
        cost_function = binary_crossentropy
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration,
        )
        # Mock initialisation of parameters
        nn.parameters = {
            "W1": np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
            "b1": np.array([[0.0], [0.0]]),
            "W2": np.array([[-0.32768579, -0.32932067]]),
            "b2": np.array([[0.0]]),
        }
        nn.grads = {
            "dW1": np.array([[0.00202345, 0.03205639], [0.00117533, 0.01911367]]),
            "db1": np.array([[0.03667271], [0.02173073]]),
            "dW2": np.array([[-0.11191426, -0.06598652]]),
            "db2": np.array([[-0.20929858]]),
        }
        # Save
        nn.save("test_save")
        # Check file exists
        self.assertTrue(os.path.isfile("test_save.pkl"))
        # Check file content
        with open("test_save.pkl", "rb") as f:
            saved_nn = pickle.load(f)
        self.assertEqual(saved_nn.input_size, input_size)
        self.assertEqual(saved_nn.layers, layers)
        self.assertEqual(saved_nn.learning_rate, learning_rate)
        self.assertEqual(saved_nn.cost_function, cost_function)
        self.assertEqual(saved_nn.verbose, verbose)
        self.assertEqual(saved_nn.verbose_iteration, verbose_iteration)
        assert np.allclose(saved_nn.parameters["W1"], nn.parameters["W1"])
        assert np.allclose(saved_nn.parameters["b1"], nn.parameters["b1"])
        assert np.allclose(saved_nn.parameters["W2"], nn.parameters["W2"])
        assert np.allclose(saved_nn.parameters["b2"], nn.parameters["b2"])
        assert np.allclose(saved_nn.grads["dW1"], nn.grads["dW1"])
        assert np.allclose(saved_nn.grads["db1"], nn.grads["db1"])
        assert np.allclose(saved_nn.grads["dW2"], nn.grads["dW2"])
        assert np.allclose(saved_nn.grads["db2"], nn.grads["db2"])

        # Remove file
        os.remove("test_save.pkl")

    def test_load(self):
        """
        Test Neural Network load
        """
        _log_test_title("Neural Network load", logger)

        input_size = 2
        # Use lineal function to avoid mocking activation functions
        layers = [(2, sigmoid), (1, sigmoid)]
        learning_rate = 0.1
        cost_function = binary_crossentropy
        verbose = True
        verbose_iteration = 100

        # Create Neural Network
        nn = NeuralNetwork(
            input_size=input_size,
            layers=layers,
            cost_function=cost_function,
            learning_rate=learning_rate,
            verbose=verbose,
            verbose_iteration=verbose_iteration,
        )
        # Mock initialisation of parameters
        nn.parameters = {
            "W1": np.array([[1.11667209, 0.5426583], [-0.33196852, 0.38364789]]),
            "b1": np.array([[0.0], [0.0]]),
            "W2": np.array([[-0.32768579, -0.32932067]]),
            "b2": np.array([[0.0]]),
        }
        nn.grads = {
            "dW1": np.array([[0.00202345, 0.03205639], [0.00117533, 0.01911367]]),
            "db1": np.array([[0.03667271], [0.02173073]]),
            "dW2": np.array([[-0.11191426, -0.06598652]]),
            "db2": np.array([[-0.20929858]]),
        }
        # Save
        Path("test_load").parent.mkdir(parents=True, exist_ok=True)
        with open("test_load.pkl", "wb") as file:
            pickle.dump(nn, file)
        # Check file exists
        self.assertTrue(os.path.isfile("test_load.pkl"))

        # Load
        saved_nn = NeuralNetwork.load("test_load")

        # Check file content
        self.assertEqual(saved_nn.input_size, input_size)
        self.assertEqual(saved_nn.layers, layers)
        self.assertEqual(saved_nn.learning_rate, learning_rate)
        self.assertEqual(saved_nn.cost_function, cost_function)
        self.assertEqual(saved_nn.verbose, verbose)
        self.assertEqual(saved_nn.verbose_iteration, verbose_iteration)
        assert np.allclose(saved_nn.parameters["W1"], nn.parameters["W1"])
        assert np.allclose(saved_nn.parameters["b1"], nn.parameters["b1"])
        assert np.allclose(saved_nn.parameters["W2"], nn.parameters["W2"])
        assert np.allclose(saved_nn.parameters["b2"], nn.parameters["b2"])
        assert np.allclose(saved_nn.grads["dW1"], nn.grads["dW1"])
        assert np.allclose(saved_nn.grads["db1"], nn.grads["db1"])
        assert np.allclose(saved_nn.grads["dW2"], nn.grads["dW2"])
        assert np.allclose(saved_nn.grads["db2"], nn.grads["db2"])

        # Remove file
        os.remove("test_load.pkl")


def load_data(path):
    train_dataset = h5py.File(path + "data/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File(path + "data/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


class NeuralNetworkIntegrationTests(unittest.TestCase):
    """
    Integration tests for Neural Network
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up all tests
        """
        super(NeuralNetworkIntegrationTests, cls).setUpClass()

    def setUp(self):
        """
        Set up all tests
        """
        np.random.seed(1)

    def test_train_evaluate_binary_classification(self):
        """
        Test the training and evaluation of Neural Network
        """
        _log_test_title(
            "Test the training and evaluation of Neural Network for Binary Classification",
            logger,
        )

        # Load data
        current_path = str(Path(os.path.dirname(os.path.realpath(__file__))))
        train_x_orig, train_y, test_x_orig, test_y, _ = load_data(current_path + "/")

        # Reshape the training and test examples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten / 255.0
        test_x = test_x_flatten / 255.0

        nn_model = NeuralNetwork(
            input_size=train_x.shape[0],
            layers=[(5, relu), (1, sigmoid)],
            learning_rate=0.0075,
            cost_function=binary_crossentropy,
            verbose=False,
        )
        nn_model.fit(train_x, train_y, epochs=5, validation_data=(test_x, test_y))
        metrics = nn_model.evaluate(train_x, train_y)
        self.assertEqual(metrics["accuracy"], 0.64)
        self.assertEqual(metrics["precision"], 0.25)
        self.assertEqual(metrics["recall"], 0.03)
        self.assertEqual(metrics["f1"], 0.05)
        metrics = nn_model.evaluate(test_x, test_y)
        self.assertEqual(metrics["accuracy"], 0.36)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.03)
        self.assertEqual(metrics["f1"], 0.06)

    def test_train_evaluate_multi_classification(self):
        """
        Test the training and evaluation of Neural Network
        """
        _log_test_title(
            "Test the training and evaluation of Neural Network for Multi Classification",
            logger,
        )

        (x_train, y_train), (x_test, y_test) = mnist_load_data()

        # Reshape the training and test examples
        x_train_flatten = x_train.reshape(x_train.shape[0], -1)
        x_test_flatten = x_test.reshape(x_test.shape[0], -1)

        # Standardize data to have feature values between 0 and 1.
        x_train = x_train_flatten / 255.0
        x_test = x_test_flatten / 255.0

        #  shrink data size
        x_train = x_train[:].T
        y_train = y_train[:]
        x_test = x_test[:].T
        y_test = y_test[:]

        # Reshape the labels from 1D to 2D
        y_train = to_categorical(y_train).T
        y_test = to_categorical(y_test).T

        nn_model = NeuralNetwork(
            input_size=x_train.shape[0],
            layers=[(5, relu), (y_train.shape[0], softmax)],
            cost_function=categorical_crossentropy,
            learning_rate=0.05,
            verbose=False,
            verbose_iteration=10,
        )
        nn_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
        metrics = nn_model.evaluate(x_train, y_train)
        self.assertEqual(metrics["accuracy"], 0.12)
        self.assertEqual(metrics["precision"], 0.09)
        self.assertEqual(metrics["recall"], 0.13)
        self.assertEqual(metrics["f1"], 0.07)
        metrics = nn_model.evaluate(x_test, y_test)
        self.assertEqual(metrics["accuracy"], 0.13)
        self.assertEqual(metrics["precision"], 0.08)
        self.assertEqual(metrics["recall"], 0.14)
        self.assertEqual(metrics["f1"], 0.07)
