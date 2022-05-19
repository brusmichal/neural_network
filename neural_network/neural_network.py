import numpy as np
from numpy.random import default_rng


def loss(predicted_y, y):
    return np.square(predicted_y - y)


def loss_derivative(predicted_y, y):
    return 2 * (predicted_y - y)


class Network(object):
    def __init__(self, neurons_per_layer):
        self.number_of_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        rng = default_rng()
        self.biases = [rng.uniform(-1 / np.sqrt(neurons_per_layer[0]), 1 / np.sqrt(neurons_per_layer[0]), size=(y, 1)) \
                       for y in neurons_per_layer[1:]]
        self.weights = [rng.uniform(-1 / np.sqrt(neurons_per_layer[0]), 1 / np.sqrt(neurons_per_layer[0]), size=(y, x)) \
                        for x, y in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]
        self.act_function = relu
        self.act_function_der = relu_derivative

    def feedforward(self, a):
        a = np.reshape(a, (-1, 1))
        for b, w in zip(self.biases, self.weights):
            a = self.act_function(np.dot(w, a) + b)
        return a

    def train_with_sgd(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            rng = default_rng()
            rng.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch: {i} / {epochs} Accuracy: {self.evaluate(test_data)} / {n_test}")
                #print(f"Epoch: {i} / {epochs} MSE: {self.mean_loss(test_data)} / {n_test}")
            else:
                print(f"Epoch {i} / {epochs} completed.")

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + d_nb for nb, d_nb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + d_nw for nw, d_nw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = np.reshape(x, (-1, 1))
        activations = [np.reshape(x, (-1, 1))]
        z_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)
            z_vector = np.dot(w, activation) + b
            z_vectors.append(z_vector)
            activation = self.act_function(z_vector)
            activations.append(activation)
        delta = loss_derivative(activations[-1], y) * self.act_function_der(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))

        for i in range(2, self.number_of_layers):
            z_vector = z_vectors[-i]
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * self.act_function_der(z_vector)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
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
