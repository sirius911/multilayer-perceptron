from tqdm import tqdm
import numpy as np
from srcs.metrics import f1_score_

def category_to_bool(arr:np.ndarray):
    res = np.zeros(len(arr))
    for idx, el in enumerate(arr):
        res[idx] = np.argmax(el)
    return res

def format_all(arr):
    """
    
    """
    tabl = arr[0].copy()
    result = []
    max = tabl.max()
    for el in tabl:
        if el>= max:
            result.append(1.0)
        else:
            result.append(0.0)
    return np.array(result)

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        historic_loss = []
        historic_f1 = []
        # training loop
        for i in tqdm(range(epochs), leave=False):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            
            #calculate f1 score
            out = self.predict(x_train)
            y_hat = category_to_bool(np.array(out).reshape(len(out),2))
            f1 = f1_score_(category_to_bool(y_train), y_hat)
            historic_f1.append(f1)
            # calculate average error on all samples
            err /= samples
            historic_loss.append(err)
        
        return historic_loss, historic_f1