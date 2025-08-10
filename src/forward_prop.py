# forward_prop.py

import numpy as np

def get_predictions(A):
    return np.argmax(A, 0)

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_z = np.exp(Z - np.max(Z, axis = 0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_prop(X, params=[]):
    result = []
    for index in range(len(params)):
        if index == 0:
            A = X
        Z = params[index][0].dot(A) + params[index][1]
        if index == len(params) - 1:
            A = softmax(Z)
        else:
            A = relu(Z)
        result.append([Z, A])
    return result
