import numpy as np
from numpy.random import default_rng


class Network(object):
    def __init__(self, neurons_per_layer):
        self.number_of_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.rng = default_rng()
        inv_sqrt_first_layer = np.sqrt(neurons_per_layer[0])
        self.biases = [
            self.rng.uniform(-inv_sqrt_first_layer, inv_sqrt_first_layer, size=(y, 1)) for y in neurons_per_layer
        ]
        self.weights = [
            self.rng.uniform(-inv_sqrt_first_layer, inv_sqrt_first_layer, size=(y, x))
            for x, y in zip(reversed(neurons_per_layer), neurons_per_layer)
        ]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train_with_sgd(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data is not None:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            self.rng.shuffle(training_data)
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

        for pixels, label in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(pixels, label)
            nabla_b = np.array([nb + d_nb for nb, d_nb in zip(nabla_b, delta_nabla_b)])
            nabla_w = np.array([nw + d_nw for nw, d_nw in zip(nabla_w, delta_nabla_w)])
            print(f'LABEL: {label}')
        self.biases = np.array([b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)])
        self.weights = np.array([w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)])

    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        print(f'LEN X ===== {np.size(x)}')
        activations = [x]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            print(f'LEN B ===== {np.size(b)}')
            print(f'LEN W ===== {np.size(w)}')
            z_vector = np.dot(w, activation) + b
            z_vectors.append(z_vector)
            activation = relu(z_vector)
            activations.append(activation)

        delta = (activations[-1] - y) * relu_derivative(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.number_of_layers):
            z_vector = z_vectors[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * relu_derivative(z_vector)
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, np.transpose(np.array(activations[-layer - 1])))
# tu wyżej jest gdzieś błąd, próbowałem użyć np array zamiast [] i nadal nie działa
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x
