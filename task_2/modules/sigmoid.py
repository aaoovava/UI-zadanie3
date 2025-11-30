import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, grad_out):
        return grad_out * (self.out * (1 - self.out))