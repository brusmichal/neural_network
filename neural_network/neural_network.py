import numpy as np
from numpy.random import default_rng


class Network(object):
    def __init__(self, neurons_per_layer):
        self.number_of_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        rng = default_rng()
        self.biases = {[rng.uniform(-1 / np.sqrt(neurons_per_layer[0]), 1 / np.sqrt(neurons_per_layer[0]), size=(y, 1)) \
                        for y in neurons_per_layer[1:]]}
        self.weights = [rng.uniform(-1 / np.sqrt(neurons_per_layer[0]), 1 / np.sqrt(neurons_per_layer[0]), size=(x, y)) \
                        for x, y in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(test_data)
        for i in range(epochs):
            rng = default_rng()
            rng.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {i} / {epochs}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {i} / {epochs} completed.")

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x.y)
            nabla_b = [nb + d_nb for nb, d_nb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + d_nw for nw, d_nw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return max(0.0, x)
