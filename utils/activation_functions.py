import numpy as np

def sigmoid(x, deriv=False):
    """
    Sigmoid activation function
    """
    # Prevent overflow
    x = np.clip(x, -709, 709)

    # Sigmoid
    x = 1 / (1 + np.exp(-x))

    # Partial derivative of sigmoid
    if deriv:
        return np.multiply(x, 1 - x)

    return x

def tanh(x, deriv=False):
    """
    Tanh activation function
    """
    # Tanh
    x = np.tanh(x)

    # Partial derivative of tanh
    if deriv:
        return 1 - np.power(x, 2)

    return x

def relu(x, deriv=False):
    """
    ReLU activation function
    """
    # Partial derivative of ReLU
    if deriv:
        return (x > 0).astype(float)

    # ReLU
    return np.maximum( 0, x )

def softmax(x, deriv=False):
    """
    Softmax activation function
    """

    # Softmax
    e_x = np.exp(x - np.max(x, axis=1, keepdims = True))
    x = e_x / np.sum(e_x, axis = 1, keepdims = True)

    # Partial derivative of softmax
    if deriv:
        return np.ones(x.shape)

    return x
