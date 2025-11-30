import json
import matplotlib.pyplot as plt
import numpy as np
from model import Model
from modules.linear import Linear
from modules.tanh import Tanh
from modules.sigmoid import Sigmoid
from optimizers.opt import OPT
from optimizers.opt_momentum import OPTMomentum
from mse_loss import MSELoss
from modules.re_lu import ReLU



def load_config(config_path='config.json'):
    with open(config_path) as f:
        return json.load(f)

def get_dataset(dataset_name):
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    if dataset_name == "AND":
        Y = np.array([[0], [0], [0], [1]], dtype=np.float32)

    elif dataset_name == "OR":
        Y = np.array([[0], [1], [1], [1]], dtype=np.float32)

    elif dataset_name == "XOR":
        Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    return X, Y

def build_model(input_dm, output_dm, hidden_layers):
    modules = []
    prev_dm = input_dm

    for h in hidden_layers:
        modules.append(Linear(prev_dm, h))
        modules.append(Tanh())
        prev_dm = h

    modules.append(Linear(prev_dm, output_dm))
    modules.append(Sigmoid())

    return Model(modules)


def main():
    config = load_config()

    X, Y = get_dataset(config['dataset'])
    hidden_layers = config['hidden_layers']
    model = build_model(input_dm=2, output_dm=1, hidden_layers=hidden_layers)
    loss_func = MSELoss()
    optimizer_config = config['optimizer']

    if optimizer_config['type'] == "OPT":
        optimizer = OPT(learning_rate=optimizer_config['learning_rate'])
    elif optimizer_config['type'] == "OPTMomentum":
        optimizer = OPTMomentum(learning_rate=optimizer_config['learning_rate'], momentum=optimizer_config['momentum'])
    else:
        raise ValueError("Bad configuration of optimizer")

    losses_history = []
    for epoch in range(config['epochs']):
        y_pred = model.forward(X)
        loss = loss_func.forward(Y, y_pred)
        grad_out = loss_func.backward()
        model.backward(grad_out=grad_out)
        optimizer.step(model)
        losses_history.append(loss)

        if epoch % 10 == 0:
            print("Epoch", epoch, "loss", loss)

    plt.plot(losses_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training loss for {config['dataset']} problem with {config['optimizer']['type']} optimizer")
    plt.show()

    y_pred = model.forward(X)
    print("test data", X, "y_pred", y_pred, "y_true", Y)

if __name__ == "__main__":
    main()