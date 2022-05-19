from mnist8 import MnistSet8
from mnist28 import DEFAULT_MNIST28
from neural_network import Network


def main():
    mnist = MnistSet8()
    training_data = mnist.get_training_set()
    # validation_data = mnist.get_validating_set()
    test_data = mnist.get_testing_set()

    network = Network([64, 15, 15, 10])
    network.train_with_sgd(training_data, 30, 10, 3.0, test_data=test_data)


if __name__ == '__main__':
    main()
