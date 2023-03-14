import datetime
import os
import numpy as np
from tqdm import tqdm
from .common import colors

class Network:
    def __init__(self):
        self.network = [] ## layers
        self.architecture = [] ## mapping input neurons --> output neurons
        self.file = "Not Saved"
        self.epochs = 100
        self.loss = []
        self.accuracy = []

    def add(self, layer):
        """
        Add layers to the network
        """
        self.network.append(layer)
            
    def _compile(self, data):
        """
        Initialize model architecture
        """
        input_shape = data.shape[1]
        for idx, layer in enumerate(self.network):
            self.architecture.append({'input_dim':input_shape, 
                                    'output_dim':self.network[idx].neurons,
                                    'activation':layer.act_name})
            layer.compile(input_dim=input_shape)
            input_shape = self.network[idx].neurons
        return self
    
    def _forwardprop(self, data, save=True):
        """
        Performs one full forward pass through network
        """
        A_curr = data  # current activation result

        # iterate over layers Weight and bias
        for layer in self.network:
            # calculate forward propagation for specific layer
            # save the ouput in A_curr and transfer it to the next layer
            A_curr = layer.forward(A_curr, save)
        return A_curr
    
    def _backprop(self, predicted, actual):
        """
        Performs one full backward pass through network
        """
        num_samples = len(actual)

        # calculate loss derivative of our algorithm
        dscores = predicted
        dscores[range(num_samples), actual] -= 1
        dscores /= num_samples

        dA_curr = dscores
        for layer in reversed(self.network):
            # calculate backward propagation for specific layer
            dA_curr = layer.backward(dA_curr)
            
    def _update(self, lr=0.01):
        """
        Update the model parameters --> lr * gradient
        """
        for layer in self.network:
            # update layer Weights and bias
            layer.update(lr)

    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        # for each sample get the index of the maximum value and compare it to the actual set
        # then compute the mean of this False/True array
        return float(np.mean(np.argmax(predicted, axis=1)==actual))
    
    def _calculate_loss(self, predicted: np.ndarray, actual: np.ndarray):
        """
        Calculate cross-entropy loss after each iteration
        """
        samples = len(actual)
        temp = predicted[range(samples), actual.astype(int)]
        temp[temp <= 0] = 1e-15
        correct_logprobs = -np.log(temp)
        data_loss = np.sum(correct_logprobs) / samples
        return float(data_loss)
    
    def train(self, X_train, y_train, epochs, verbose = False):
        """
        Train the model
        """
        if epochs > 10:
            pas = epochs / 10
        else:
            pas = 1
        loop = range(epochs) if verbose else tqdm(range(epochs), leave=False, colour='green')

        for i in loop:
                yhat = self._forwardprop(X_train, save=True)
                accuracy = self._get_accuracy(predicted=yhat, actual=y_train)
                self.accuracy.append(accuracy)
                loss = self._calculate_loss(predicted=yhat, actual=y_train)
                self.loss.append(loss)
                self._backprop(predicted=yhat, actual=y_train)
                self._update()
                if verbose and i%pas == 0:
                    print(f"epoch {i}/{epochs} - loss:{loss:} accuracy : {accuracy}")
        return loss, accuracy
       
    def predict(self, X):
        y_hat = self._forwardprop(X, save=False)
        return y_hat

    def get_cross_entropy(self):
        """
            return the last calcul of Entropy
        """
        if len(self.loss) > 0:
            return self.loss[-1]
        return None

    def __str__(self):
        file_name = f"models/{self.file}"
        timestamp_modif = os.path.getmtime(file_name)
        date_modif = datetime.datetime.fromtimestamp(timestamp_modif)   
        res = colors.yellow + str(self.file) + colors.reset + " with "
        for layer in self.network:
            res = res + layer.__str__() + "\t"
        res = res + "\n\tlast training : " + colors.yellow + date_modif.strftime("%Y-%m-%d %H:%M:%S") + colors.reset
        res = res + "\n\tCross-Entropy : " + colors.blue + str(self.get_cross_entropy()) + colors.reset
        return res