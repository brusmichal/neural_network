# Author: Jakub Mazurkiewicz
from itertools import islice
import sys

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mnist_base import MnistSetBase

class MnistSet8(MnistSetBase):
    """
    The MNIST set from `sklearn`.
    """
    def __init__(self):
        super().__init__()
        self.digits = load_digits()
        self._split_sets()

    def _split_sets(self):
        self.training_set, self.testing_set = train_test_split(self.digits.data)
        self.training_set, self.validating_set = train_test_split(self.training_set)

    def images(self):
        for img in self.digits.images:
            yield img

    def flattened_images(self):
        for img in self.digits.data:
            yield img

def main():
    img_count = int(sys.argv[1]) if len(sys.argv) >= 2 else 10
    mnist = MnistSet8()
    print(mnist)

    plt.gray()
    for img in islice(mnist.images(), img_count):
        plt.matshow(img)
    plt.show()

if __name__ == '__main__':
    main()
