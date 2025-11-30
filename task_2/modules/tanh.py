import numpy as np

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1 - self.out**2)