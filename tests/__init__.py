'''
Unit tests for DL Workshop
'''

import unittest

import os
import sys
import logging

from tests.activation_functions_unittests import ActivationFunctionTests

assert ActivationFunctionTests

sys.path.append(os.getcwd())

def _log_test_title(title, logger):
    line = "=" * len(title)
    logger.info("\n\n%s\n%s\n%s" % (line, title, line))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()