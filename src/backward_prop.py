# backward_prop.py

import numpy as np

def deriv_relu(Z):
    return Z > 0

def deriv_weight_and_bias(m, dz, A):
    dw = (1 / m) * dz.dot(A.T)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
    return [dw, db]

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(X, Y, params=[], weights=[]):
    results = []
    dZ = params[len(params) - 1][1] - one_hot(Y)
    for index in range(len(params) - 2, -1, -1):
        A = params[index][1]
        Z = params[index][0]
        W = weights[index + 1]
        results.append(deriv_weight_and_bias(Y.size, dZ, A))
        dZ = W.T.dot(dZ) * deriv_relu(Z)
    results.append(deriv_weight_and_bias(Y.size, dZ, X))
    return results

        # A_current = params[index][1] if index == 1 else X
        # A = params[index - 1][1] if index == 1 else X
        # W = weights[index + 1][0] if index < last_index else None
        # Z = params[index + 1][0] if index < last_index else None
        # dz = (A_current - one_hot(Y)) if index == last_index else (W.T.dot(A_current) * deriv_relu(Z))
        # dW, db = deriv_weight_and_bias(y_size, dz, A)
        # result.append([dW, db])
    # return results

    # A2 = params[1][1]
    # A1 = params[0][1]
    # W2 = weights[1]
    # Z1 = params[0][0]

    # dz2 = A2 - one_hot(Y)
    # dw2, db2 = deriv_weight_and_bias(Y.size, dz2, A1)
    # dz1 = W2.T.dot(dz2) * deriv_relu(Z1)
    # dw1, db1 = deriv_weight_and_bias(Y.size, dz1, X)
    # return [[dw2, db2], [dw1, db1]]