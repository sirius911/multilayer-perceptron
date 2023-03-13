import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

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