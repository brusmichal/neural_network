from mnist8 import MnistSet8
from mnist28 import DEFAULT_MNIST28
from neural_network import Network

def main():
    mnist = MnistSet8()
    training_data = mnist.get_training_images()
    validation_data = mnist.get_validating_images()
    test_data = mnist.get_testing_images()

    network = Network([64, 15, 15, 10])
    network.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)
