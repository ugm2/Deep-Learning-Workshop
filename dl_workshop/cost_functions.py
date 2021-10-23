import numpy as np

def binary_crossentropy(y_true, y_pred, deriv=False, eps=1e-12):
    """
    Binary Crossentropy
    """
    if deriv:
        # Prevent overflow
        outputs = np.clip(y_pred, eps, 1 - eps)
        divisor = np.maximum(outputs * (1 - outputs), eps)
        return (y_pred - y_true) / divisor
    return -np.sum(np.multiply(y_true, np.log(y_pred)) + np.multiply((1 - y_true), np.log(1 - y_pred))) / y_true.shape[1]

def categorical_crossentropy(y_true, y_pred, deriv=False, epsilon=1e-12):
    """
    Categorical Crossentropy
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    if deriv:
        return y_pred - y_true
    else:
        return -np.sum(np.multiply(y_true, np.log(y_pred))) / y_true.shape[1]