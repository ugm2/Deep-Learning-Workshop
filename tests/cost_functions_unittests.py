'''
Unit tests for Cost Functions
'''

import unittest
import numpy as np
import logging
import os

from dl_workshop.cost_functions import (
    binary_crossentropy,
    categorical_crossentropy
)

from tests.utils import _log_test_title

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger("Cost Functions unittest")

class CostFunctionTests(unittest.TestCase):
    '''
    Unit tests for Cost Functions
    '''
    def test_binary_crossentropy(self):
        '''
        Test binary_crossentropy
        '''
        _log_test_title("Test Binary Cross Entropy", logger)
        y_true = np.array([[0, 1, 0, 1]])
        y_pred = np.array([[0.1, 0.9, 0.1, 0.9]])
        loss = binary_crossentropy(y_true, y_pred)
        logger.debug("Loss: %s", loss)
        self.assertAlmostEqual(loss,  0.10536051565782628)

        # Test derivative
        y_true = np.array([[0, 1, 0, 1]])
        y_pred = np.array([[0.1, 0.9, 0.1, 0.9]])
        loss = binary_crossentropy(y_true, y_pred, deriv=True)
        logger.debug("Loss: %s", loss)
        assert np.allclose(loss, [[ 1.11111111, -1.11111111, 1.11111111, -1.11111111]])

    def test_categorical_crossentropy(self):
        '''
        Test categorical_crossentropy
        '''
        _log_test_title("Test Categorical Cross Entropy", logger)
        y_true = np.array([[0.0, 1.0],
                           [0.0, 0.0],
                           [1.0, 0.0]])
        y_pred = np.array([[0.0, 0.9],
                           [0.1, 0.2],
                           [0.9, 0.0]])
        loss = categorical_crossentropy(y_true, y_pred)
        logger.debug("Loss: %s", loss)
        self.assertAlmostEqual(loss, 0.10536051565782628)

        # Test derivative
        y_true = np.array([[0, 1, 0, 1]])
        y_pred = np.array([[0.1, 0.9, 0.1, 0.9]])
        loss = categorical_crossentropy(y_true, y_pred, deriv=True)
        logger.debug("Loss: %s", loss)
        assert np.allclose(loss, [[ 1.11111111, -1.11111111, 1.11111111, -1.11111111]])