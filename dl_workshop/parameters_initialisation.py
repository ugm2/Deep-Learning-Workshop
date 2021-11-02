"""Implementation of Parameter Initialisation."""
import numpy as np


def zero_initialisation(dimensions):
    """Return zero vectors for W and b."""
    parameters = {}
    for d in range(1, len(dimensions)):
        parameters["W" + str(d)] = np.zeros((dimensions[d], dimensions[d - 1]))
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))

    return {"parameters": parameters, "grads": {}}


def random_initialisation(dimensions):
    """Return random vectors for W and b."""
    parameters = {}
    for d in range(1, len(dimensions)):
        parameters["W" + str(d)] = (
            np.random.randn(dimensions[d], dimensions[d - 1]) * 0.01
        )
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    return {"parameters": parameters, "grads": {}}


def he_initialisation(dimensions, distribution="normal"):
    """Return He initialisation for W and b."""
    parameters = {}
    for d in range(1, len(dimensions)):
        if distribution == "normal":
            limit = np.sqrt(2 / dimensions[d - 1])
            parameters["W" + str(d)] = np.random.normal(
                0.0, limit, size=(dimensions[d], dimensions[d - 1])
            )
        elif distribution == "uniform":
            limit = np.sqrt(6 / dimensions[d - 1])
            parameters["W" + str(d)] = np.random.uniform(
                low=-limit, high=limit, size=(dimensions[d], dimensions[d - 1])
            )
        else:
            raise ValueError("Distribution must be either normal or uniform")
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    return {"parameters": parameters, "grads": {}}


def xavier_initialisation(dimensions, distribution="uniform"):
    """Return Xavier initialisation for W and b."""
    parameters = {}
    for d in range(1, len(dimensions)):
        if distribution == "uniform":
            limit = np.sqrt(6 / (dimensions[d - 1] + dimensions[d]))
            parameters["W" + str(d)] = np.random.uniform(
                low=-limit, high=limit, size=(dimensions[d], dimensions[d - 1])
            )
        elif distribution == "normal":
            limit = np.sqrt(2 / (dimensions[d - 1] + dimensions[d]))
            parameters["W" + str(d)] = np.random.normal(
                0.0, limit, size=(dimensions[d], dimensions[d - 1])
            )
        else:
            raise ValueError("Distribution must be either normal or uniform")
        parameters["b" + str(d)] = np.zeros((dimensions[d], 1))
    return {"parameters": parameters, "grads": {}}
