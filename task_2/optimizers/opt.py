class OPT:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def step(self, model):
        for p, g in model.params_and_grads():
            p -= self.learning_rate * g