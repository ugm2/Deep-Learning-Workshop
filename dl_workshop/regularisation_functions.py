"""Implementation of Regularisation Functions"""
import numpy as np

def l1(weights, lmbda, m, backward=False):
    """L1 Regularisation method"""
    if backward:
        return [lmbda * np.sign(w) / (2*m) for w in weights]
    return sum([np.sum(np.abs(w)) for w in weights]) * (lmbda / (2*m))
        
def l2(weights, lmbda, m, backward=False):
    """L2 Regularisation method"""
    if backward:
        return [lmbda * w / m for w in weights]
    return sum([np.sum(np.square(w)) for w in weights]) * (lmbda / (2*m))