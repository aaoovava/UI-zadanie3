class Model:
    def __init__(self, modules):
        self.modules = modules

    def forward(self, X):
        for module in self.modules:
            X = module.forward(X)
        return X

    def backward(self, grad_out):
        for module in reversed(self.modules):
            grad_out = module.backward(grad_out)
        return grad_out

    def params_and_grads(self):
        for module in self.modules:
            if hasattr(module, "params") and hasattr(module, "grads"):
                for p, g in zip(module.params(), module.grads()):
                    yield p, g