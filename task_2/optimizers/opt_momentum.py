import numpy as np

class OPTMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.8):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {} # (param(id), velocity)

    def step(self, model):
        for p, g in model.params_and_grads():
            param_id = id(p)
            if param_id not in self.velocities:
                self.velocities[param_id] = np.zeros_like(p)

            v = self.velocities[param_id]
            v = self.momentum * v - self.learning_rate * g
            self.velocities[param_id] = v
            p += v