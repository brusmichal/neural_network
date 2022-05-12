# Authors: Jakub Mazurkiewicz, Michał Brus
from itertools import islice
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sys

class MnistSetDeserializer:
    pass

class MnistSet:
    """
    The MNIST set: https://en.wikipedia.org/wiki/MNIST_database
    """
    def __init__(self):
        self.digits = load_digits()
        self.training_set = []
        self.validating_set = []
        self.testing_set = []
        self._split_sets()

    def _split_sets(self):
        self.training_set, self.testing_set = train_test_split(self.digits.data)
        self.training_set, self.validating_set = train_test_split(self.training_set)

    def get_training_set(self):
        return self.training_set

    def get_validating_set(self):
        return self.validating_set

    def get_testing_set(self):
        return self.training_set

    def images(self):
        for img in self.digits.images:
            yield img

    def flattened_images(self):
        for img in self.digits.data:
            yield img

    def print_info(self):
        print(f'Training set size:   {len(self.training_set)}')
        print(f'Validating set size: {len(self.validating_set)}')
        print(f'Testing set size:    {len(self.testing_set)}')

def main():
    img_count = int(sys.argv[1]) if len(sys.argv) >= 2 else 10
    mnist = MnistSet()
    mnist.print_set_info()

    plt.gray()
    for img in islice(mnist.images(), img_count):
        plt.matshow(img)
    plt.show()

if __name__ == '__main__':
    main()
