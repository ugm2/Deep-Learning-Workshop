"""Implementation of Parameter Initialisation."""
import numpy as np

def zero_initialization(dimensions):
    """Return zero vectors for W and b"""
    parameters = {}
    for d in range(1, len(dimensions)):
        parameters["W" + str(d)] = np.zeros((dimensions[d], dimensions[d-1]))
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    
    return {
        "parameters": parameters,
        "grads": {}
    }

def random_initialization(dimensions):
    """Return random vectors for W and b"""
    parameters = {}
    for d in range(1, len(dimensions)):
        parameters["W" + str(d)] = np.random.randn(dimensions[d], dimensions[d-1])
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    return {
        "parameters": parameters,
        "grads": {}
    }

def he_initialization(dimensions):
    """Return He initialization for W and b"""
    parameters = {}
    for d in range(1, len(dimensions)):
        parameters["W" + str(d)] = np.random.randn(dimensions[d], dimensions[d-1]) / np.sqrt(2/dimensions[d-1])
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    return {
        "parameters": parameters,
        "grads": {}
    }

def xavier_initialization(dimensions):
    """Return Xavier initialization for W and b"""
    parameters = {}
    for d in range(1, len(dimensions)):
        parameters["W" + str(d)] = np.random.randn(dimensions[d], dimensions[d-1]) / np.sqrt(dimensions[d-1])
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    return {
        "parameters": parameters,
        "grads": {}
    }
