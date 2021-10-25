'''
Unit tests for DL Workshop
'''

import unittest

import os
import sys
import logging

from tests.activation_functions_unittests import ActivationFunctionTests
from tests.cost_functions_unittests import CostFunctionTests

assert ActivationFunctionTests
assert CostFunctionTests

sys.path.append(os.getcwd())

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()