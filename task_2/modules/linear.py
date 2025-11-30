import numpy as np

class Linear:
    def __init__(self, in_items, out_items):
        limit = np.sqrt(6/ (in_items + out_items))
        self.W = np.random.uniform(-limit, limit, size=(in_items, out_items))
        self.b = np.zeros((1, out_items))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        grad_in = grad_out @ self.W.T
        return grad_in

    def params(self):
        return self.W, self.b

    def grads(self):
        return self.dW, self.db