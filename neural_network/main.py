from mnist8 import MnistSet8
from mnist28 import DEFAULT_MNIST28
from neural_network import Network


def main():
    mnist = MnistSet8()
    training_data = mnist.get_training_set()
    # validation_data = mnist.get_validating_set()
    test_data = mnist.get_testing_set()

    network = Network([64, 15, 10])
    network.SGD(training_data, 30, 10, 0.001, test_data=test_data)


if __name__ == '__main__':
    main()
