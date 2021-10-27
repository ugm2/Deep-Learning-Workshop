"""
Unit tests for DL Workshop
"""

import unittest

import os
import sys
import logging

from tests.activation_functions_unittests import ActivationFunctionTests
from tests.cost_functions_unittests import CostFunctionTests
from tests.logistic_regression_tests import (
    LogisticRegressionUnitTests,
    LogisticRegressionIntegrationTests,
)
from tests.neural_network_tests import (
    NeuralNetworkUnitTests,
    NeuralNetworkIntegrationTests,
)

assert ActivationFunctionTests
assert CostFunctionTests
assert LogisticRegressionUnitTests
assert LogisticRegressionIntegrationTests
assert NeuralNetworkUnitTests
assert NeuralNetworkIntegrationTests

sys.path.append(os.getcwd())

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
