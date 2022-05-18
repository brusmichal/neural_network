# Author: Jakub Mazurkiewicz
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(x) / ((np.exp(x) + 1) ** 2)

class NeuralNetwork:
    """
    The neural network.
    """
    def __init__(self):
        pass
