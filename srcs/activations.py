import numpy as np


# activation function and its derivative
def tanh(inputs):
    return np.tanh(inputs)

def tanh_prime(inputs, Z):
    return 1-np.tanh(inputs)**2

def sigmoid(inputs):
    inputs = np.clip(inputs, -500, 500)
    return 1/(1 + np.exp(-inputs))

def sigmoid_prime(inputs, Z):
    return sigmoid(inputs) * (1 - sigmoid(inputs))

def relu(inputs):
    """
    ReLU Activation Function
    """
    return np.maximum(0, inputs)

def relu_prime(dA, Z):
    """
    ReLU Derivative Function
    """
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(inputs):
    """
    Softmax Activation Function
    """
    exp_scores = np.exp(inputs)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def softmax_prime(dA, Z):
    """
    Softmax Derivative Function
    """
    return dA

act_funct = {
    "relu":relu,
    "softmax":softmax,
    "sigmoid":sigmoid,
    "tanh":tanh,
}

der_funct = {
    "relu":relu_prime,
    "softmax":softmax_prime,
    "sigmoid":sigmoid_prime,
    "tanh":tanh_prime,
}