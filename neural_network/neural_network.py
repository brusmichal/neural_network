import numpy as np
from numpy.random import default_rng


def loss(predicted_y, expected_y):
    return np.square(predicted_y - expected_y)


def loss_derivative(predicted_y, expected_y):
    return 2 * (predicted_y - expected_y)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def chunks(r, n):
    for i in range(0, len(r), n):
        yield r[i:i+n]

class Network:
    def __init__(self, neurons_per_layer):
        self.number_of_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.rng = default_rng()
        inv_sqrt_first_layer_size = 1 / np.sqrt(neurons_per_layer[0])
        self.biases = [self.rng.uniform(-inv_sqrt_first_layer_size, inv_sqrt_first_layer_size, size=(y, 1))
                       for y in neurons_per_layer[1:]]
        self.weights = [self.rng.uniform(-inv_sqrt_first_layer_size, inv_sqrt_first_layer_size, size=(y, x))
                        for x, y in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]

    def feedforward(self, a):
        a = np.reshape(a, (-1, 1))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, mini_batch_size, learning_rate):
        n = len(training_data)
        self.rng.shuffle(training_data)
        mini_batches = chunks(training_data, mini_batch_size)
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, learning_rate)

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(np.reshape(x, (-1, 1)), y)
            nabla_b = [nb + d_nb for nb, d_nb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + d_nw for nw, d_nw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (nb * learning_rate / len(mini_batch)) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (nw * learning_rate / len(mini_batch)) for w, nw in zip(self.weights, nabla_w)]

    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z_vector = np.dot(w, activation) + b
            z_vectors.append(z_vector)
            activation = sigmoid(z_vector)
            activations.append(activation)
        delta = loss_derivative(activations[-1], self.one_hot_encode(y)) * sigmoid_derivative(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))

        for i in range(2, self.number_of_layers):
            z_vector = z_vectors[-i]
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_derivative(z_vector)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
        return nabla_b, nabla_w

    def evaluate(self, data_set):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data_set]
        return sum(int(x == y) for (x, y) in results)

    def mse(self, test_data):
        err = [loss(self.feedforward(x), self.one_hot_encode(y)) for (x, y) in test_data]
        mean_err = np.mean(err)
        return mean_err

    def one_hot_encode(self, j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

