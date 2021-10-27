"""
Unit tests for Logistic Regression
"""

import pickle
import unittest
from unittest import mock
import h5py
import numpy as np
import logging
import os
from pathlib import Path

from dl_workshop.logistic_regression import LogisticRegression

from tests.utils import _log_test_title

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger("Logistic Regression tests")


class LogisticRegressionUnitTests(unittest.TestCase):
    """
    Unit tests for Logistic Regression
    """

    def test_init(self):
        """
        Test the initialization of Logistic Regression
        """
        _log_test_title("Test the initialization of Logistic Regression", logger)

        learning_rate = 0.01
        verbose = True
        lr = LogisticRegression(learning_rate, verbose)
        self.assertIsNotNone(lr)
        self.assertEqual(lr.learning_rate, learning_rate)
        self.assertEqual(lr.verbose, verbose)
        self.assertEqual(lr.w, None)
        self.assertEqual(lr.b, None)
        self.assertEqual(lr.grads, None)
        self.assertEqual(lr.costs, None)

    def test_init_params(self):
        """
        Test the initialization of Logistic Regression with parameters
        """
        _log_test_title("Test Logistic Regression init_params method", logger)

        lr = LogisticRegression(0.01, False)
        w, b = lr.init_params(dim=2)
        self.assertIsNotNone(lr)
        self.assertEqual(w.shape, (2, 1))
        assert np.allclose(w, np.zeros((2, 1)))
        self.assertEqual(b, 0.0)

    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.init_params")
    @mock.patch("dl_workshop.activation_functions.sigmoid")
    @mock.patch("dl_workshop.cost_functions.binary_crossentropy")
    def test_propagate(self, mock_cost_function, sigmoid_mock, init_params_mock):
        """
        Test the propagation of Logistic Regression
        """
        _log_test_title("Test the propagation of Logistic Regression", logger)

        # Mock the init_params method
        init_params_mock.return_value = (np.zeros((2, 1)), 0.0)
        # Mock the sigmoid method
        sigmoid_mock.return_value = np.array([[0.5, 0.5]])
        # Mock the cost function
        mock_cost_function.return_value = 0.6931471805599453

        lr = LogisticRegression(0.01, False)
        w, b = lr.init_params(dim=2)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0]])
        lr.w = w
        lr.b = b
        grads, cost = lr.propagate(X, y)
        self.assertIsNotNone(lr)
        self.assertEqual(lr.w.shape, (2, 1))
        assert np.allclose(lr.w, np.array([[0.0], [0.0]]))
        self.assertEqual(lr.b, 0.0)
        self.assertIn("dw", grads)
        self.assertIn("db", grads)
        self.assertEqual(grads["dw"].shape, (2, 1))
        assert np.allclose(grads["dw"], np.array([[0.25], [0.25]]))
        self.assertEqual(grads["db"], 0.0)
        self.assertEqual(cost, 0.6931471805599453)

    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.init_params")
    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.propagate")
    def test_optimize(self, mock_propagate, init_params_mock):
        """
        Test the optimization of Logistic Regression
        """
        _log_test_title("Test the optimization of Logistic Regression", logger)

        # Mock the init_params method
        init_params_mock.return_value = (np.zeros((2, 1)), 0.0)
        # Mock the propagate method
        mock_propagate.return_value = (
            {"dw": np.array([[0.25], [0.25]]), "db": 0.0},
            0.6931471805599453,
        )

        lr = LogisticRegression(0.01, False)
        w, b = lr.init_params(dim=2)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0]])
        lr.w = w
        lr.b = b
        grads, costs = lr.optimize(X, y, epochs=1)
        assert np.allclose(lr.w, np.array([[-0.0025], [-0.0025]]))
        self.assertEqual(lr.b, 0.0)
        self.assertEqual(costs[0], 0.6931471805599453)
        self.assertEqual(grads["dw"].shape, (2, 1))
        assert np.allclose(grads["dw"], np.array([[0.25], [0.25]]))
        self.assertEqual(grads["db"], 0.0)

    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.init_params")
    @mock.patch("dl_workshop.activation_functions.sigmoid")
    def test_predict(self, mock_sigmoid, init_params_mock):
        """
        Test the prediction of Logistic Regression
        """
        _log_test_title("Test the prediction of Logistic Regression", logger)

        # Mock the init_params method
        init_params_mock.return_value = (np.zeros((2, 1)), 0.0)
        # Mock the sigmoid method
        mock_sigmoid.return_value = np.array([[0.5, 0.5]])

        lr = LogisticRegression(0.01, False)
        w, b = lr.init_params(dim=2)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0]])
        lr.w = w
        lr.b = b
        y_pred = lr.predict(X)
        self.assertEqual(y_pred.shape, (1, 2))
        assert np.allclose(y_pred, np.array([[0, 0]]))

    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.init_params")
    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.optimize")
    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.predict")
    def test_fit(self, mock_predict, mock_optimize, init_params_mock):
        """
        Test the fit of Logistic Regression
        """
        import io
        import sys

        _log_test_title("Test the fit of Logistic Regression", logger)

        # Mock the init_params method
        init_params_mock.return_value = (np.zeros((2, 1)), 0.0)
        # Mock the optimize method
        mock_optimize.return_value = (
            {"dw": np.array([[0.25], [0.25]]), "db": 0.0},
            [0.6931471805599453],
        )
        # Mock the predict method
        mock_predict.return_value = np.array([[0, 0]])

        lr = LogisticRegression(0.01, True)
        w, b = lr.init_params(dim=2)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0]])
        lr.w = w
        lr.b = b
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        lr.fit(X, y, epochs=1)
        sys.stdout = sys.__stdout__
        self.assertEqual(lr.costs[0], 0.6931471805599453)
        self.assertEqual(lr.grads["dw"].shape, (2, 1))
        assert np.allclose(lr.grads["dw"], np.array([[0.25], [0.25]]))
        self.assertEqual(lr.grads["db"], 0.0)
        self.assertEqual(capturedOutput.getvalue(), "train accuracy: 50.0 %\n")

    @mock.patch("dl_workshop.logistic_regression.LogisticRegression.predict")
    def test_evaluate(self, mock_predict):
        """
        Test the evaluation of Logistic Regression
        """
        _log_test_title("Test the evaluation of Logistic Regression", logger)

        # Mock the predict method
        mock_predict.return_value = np.array([[0, 0, 1]])

        lr = LogisticRegression(0.01, True)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([[1, 0, 1]])
        metrics = lr.evaluate(X, y)
        accuracy = metrics["accuracy"]
        self.assertEqual(accuracy, 0.67)
        precision = metrics["precision"]
        self.assertEqual(precision, 0.75)
        recall = metrics["recall"]
        self.assertEqual(recall, 0.75)
        f1 = metrics["f1"]
        self.assertEqual(f1, 0.67)

    def test_save(self):
        """
        Test the save of Logistic Regression
        """
        _log_test_title("Test the save of Logistic Regression", logger)

        lr = LogisticRegression(0.01, True)
        lr.w = np.array([[0.25], [0.25]])
        lr.b = 0.0
        lr.grads = {"dw": np.array([[0.25], [0.25]]), "db": 0.0}
        lr.costs = [0.6931471805599453]
        lr.save("test_save")

        # Check file exists
        self.assertTrue(os.path.isfile("test_save.pkl"))

        # Check file content by loading pickle
        with open("test_save.pkl", "rb") as file:
            saved_lr = pickle.load(file)
        self.assertEqual(saved_lr.w.shape, (2, 1))
        assert np.allclose(saved_lr.w, np.array([[0.25], [0.25]]))
        self.assertEqual(saved_lr.b, 0.0)
        self.assertEqual(saved_lr.grads["dw"].shape, (2, 1))
        assert np.allclose(saved_lr.grads["dw"], np.array([[0.25], [0.25]]))
        self.assertEqual(saved_lr.grads["db"], 0.0)
        self.assertEqual(saved_lr.costs[0], 0.6931471805599453)

        # Remove file
        os.remove("test_save.pkl")

    def test_load(self):
        """
        Test the load of Logistic Regression
        """
        _log_test_title("Test the load of Logistic Regression", logger)

        lr = LogisticRegression(0.01, True)
        lr.w = np.array([[0.25], [0.25]])
        lr.b = 0.0
        lr.grads = {"dw": np.array([[0.25], [0.25]]), "db": 0.0}
        lr.costs = [0.6931471805599453]
        Path("test_load").parent.mkdir(parents=True, exist_ok=True)
        with open("test_load.pkl", "wb") as file:
            pickle.dump(lr, file)

        # Check file exists
        self.assertTrue(os.path.isfile("test_load.pkl"))

        # Check file content by loading pickle
        saved_lr = LogisticRegression.load("test_load")
        self.assertEqual(saved_lr.w.shape, (2, 1))
        assert np.allclose(saved_lr.w, np.array([[0.25], [0.25]]))
        self.assertEqual(saved_lr.b, 0.0)
        self.assertEqual(saved_lr.grads["dw"].shape, (2, 1))
        assert np.allclose(saved_lr.grads["dw"], np.array([[0.25], [0.25]]))
        self.assertEqual(saved_lr.grads["db"], 0.0)
        self.assertEqual(saved_lr.costs[0], 0.6931471805599453)

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


class LogisticRegressionIntegrationTests(unittest.TestCase):
    """
    Integration tests for Logistic Regression
    """

    def test_train_evaluate(self):
        """
        Test the training and evaluation of Logistic Regression
        """
        _log_test_title(
            "Test the training and evaluation of Logistic Regression", logger
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

        # Train Logistic Regression
        lr_model = LogisticRegression(learning_rate=0.0075, verbose=False)
        lr_model.fit(train_x, train_y, epochs=400, validation_data=(test_x, test_y))

        # Check metrics
        metrics = lr_model.evaluate(train_x, train_y)
        self.assertEqual(metrics["accuracy"], 0.81)
        self.assertEqual(metrics["precision"], 0.81)
        self.assertEqual(metrics["recall"], 0.85)
        self.assertEqual(metrics["f1"], 0.81)
        metrics = lr_model.evaluate(test_x, test_y)
        self.assertEqual(metrics["accuracy"], 0.84)
        self.assertEqual(metrics["precision"], 0.84)
        self.assertEqual(metrics["recall"], 0.79)
        self.assertEqual(metrics["f1"], 0.81)
