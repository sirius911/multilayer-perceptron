import numpy as np
from tqdm import tqdm

class Network:
    def __init__(self):
        self.network = [] ## layers
        self.architecture = [] ## mapping input neurons --> output neurons
        
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
        self.loss = []
        self.accuracy = []
        if verbose:
            for i in range(epochs):
                yhat = self._forwardprop(X_train, save=True)
                accuracy = self._get_accuracy(predicted=yhat, actual=y_train)
                self.accuracy.append(accuracy)
                loss = self._calculate_loss(predicted=yhat, actual=y_train)
                self.loss.append(loss)
                self._backprop(predicted=yhat, actual=y_train)
                self._update()
                if i%10 == 0:
                    print(f"epoch {i}/{epochs} - loss:{loss:} accuracy : {accuracy}")
        else:
            for i in tqdm(range(epochs), leave=False, colour='green'):
            
                yhat = self._forwardprop(X_train, save=True)
                accuracy = self._get_accuracy(predicted=yhat, actual=y_train)
                self.accuracy.append(accuracy)
                loss = self._calculate_loss(predicted=yhat, actual=y_train)
                self.loss.append(loss)
                self._backprop(predicted=yhat, actual=y_train)
                self._update()

    def predict(self, X):
        y_hat = self._forwardprop(X, save=False)
        return y_hat