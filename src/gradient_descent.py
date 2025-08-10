import numpy as np
import accuracy_plot
import forward_prop
import backward_prop

def init_params(layers=2):
    results = []
    for i in range(layers - 1):
        W = np.random.rand(10, 784) if i == 0 else np.random.rand(10, 10)
        b = np.random.rand(10, 1) - 0.5
        results.append([W - 0.5, b])

    # Last layer
    W = np.random.rand(10, 10) - 0.5
    b = np.random.rand(10, 1) - 0.5
    results.append([W, b])
    return results

def update_params(weights_and_biases, derivates, alpha):
    results = []
    derivates.reverse()
    for index in range(len(weights_and_biases)):
        W, b = weights_and_biases[index]
        dw, db = derivates[index]
        W -= alpha * dw
        b -= alpha * db
        results.append([W, b])

    return results

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_weights(params):
    weights = []
    for param in params:
        weights.append(param[0])
    return weights

def gradient_descent(X, Y, iterations, alpha, layers=2):
    plot = accuracy_plot.AccuracyPlot()
    weights_and_biases = init_params(layers)
    for _ in range(iterations):
        params = forward_prop.forward_prop(X, weights_and_biases)
        derivaties = backward_prop.backward_prop(X, Y, params, get_weights(weights_and_biases))
        weights_and_biases = update_params(weights_and_biases, derivaties, alpha)
        plot.update(get_accuracy(forward_prop.get_predictions(params[1][1]), Y))
    plot.close()
    return weights_and_biases
