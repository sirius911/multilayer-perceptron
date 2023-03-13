# Base class
import numpy as np
from .activations import act_funct, der_funct
from .common import colors

class DenseLayer:
    def __init__(self, neurons, act_name='relu'):
        self.neurons = neurons
        self.act_name = act_name
        self.W = None
        self.b = None
        self.regul = None

    def compile(self, input_dim):
        self.input_dim = input_dim
        self.activation = act_funct[self.act_name]
        self.activation_der = der_funct[self.act_name]
        np.random.seed(99)
        self.W = np.random.uniform(low=-1,
                                    high=1,
                                    size=(self.neurons, input_dim))
        self.b = np.zeros((1, self.neurons))

    def forward(self, inputs, save):
        """
        Single Layer Forward Propagation
        """
        Z_curr = np.dot(inputs, self.W.T) + self.b
        A_curr = self.activation(inputs=Z_curr)
        if save:
            self.mI, self.mZ = inputs, Z_curr
        return A_curr

    def backward(self, dA_curr):
        """
        Single Layer Backward Propagation
        """
        dZ = self.activation_der(dA_curr, self.mZ)
        dW = np.dot(self.mI.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.W)

        self.dW, self.db = dW, db
        return dA

    def update(self, lr: float):
        if self.regul == 'l2':
            _lambda = 0.5
            m = len(self.dw)
            self.dW += (_lambda / (2 * m)) * np.sum(self.dW ** 2)

        self.W -= lr * self.dW.T
        self.b -= lr * self.db

    def __str__(self):
        return f"({colors.blue}{str(self.neurons)}{colors.reset} neurons [{colors.green}{self.act_name}{colors.reset}])"