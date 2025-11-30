import numpy as np

class ReLU:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.out = np.maximum(0, X)
        return self.out

    def backward(self, grad_out):
        return grad_out * (self.X > 0).astype(grad_out.dtype)