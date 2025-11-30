import numpy as np

class MSELoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        N = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) * (2.0 / N)