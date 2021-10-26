'''
Unit tests for DL Workshop
'''

import unittest

import os
import sys
import logging

from tests.activation_functions_unittests import ActivationFunctionTests
from tests.cost_functions_unittests import CostFunctionTests
from tests.logistic_regression_unittests import LogisticRegressionUnitTests, LogisticRegressionIntegrationTests

assert ActivationFunctionTests
assert CostFunctionTests
assert LogisticRegressionUnitTests
assert LogisticRegressionIntegrationTests

sys.path.append(os.getcwd())

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()