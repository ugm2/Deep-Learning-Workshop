import numpy as np


def binary_crossentropy(y_true, y_pred, deriv=False, eps=1e-12):
    """
    Binary Crossentropy
    """
    # Prevent overflow
    outputs = np.clip(y_pred, eps, 1 - eps)
    divisor = np.maximum(outputs * (1 - outputs), eps)

    if deriv:
        return (y_pred - y_true) / divisor
    return (
        -np.sum(
            np.multiply(y_true, np.log(outputs))
            + np.multiply((1 - y_true), np.log(1 - outputs))
        )
        / y_true.shape[1]
    )


def categorical_crossentropy(y_true, y_pred, deriv=False, eps=1e-12):
    """
    Categorical Crossentropy
    """
    # Prevent overflow
    outputs = np.clip(y_pred, eps, 1 - eps)
    divisor = np.maximum(outputs * (1 - outputs), eps)

    if deriv:
        return (y_pred - y_true) / divisor
    return -np.sum(np.multiply(y_true, np.log(outputs))) / y_true.shape[1]
