import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(W1.T, x) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(W2.T, z1) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(W3.T, z2) + b3
    y = identity_function(a3)

    return y