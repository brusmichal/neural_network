# Author: Jakub Mazurkiewicz
import sys

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mnist_base import MnistSetBase

class MnistSet8(MnistSetBase):
    """
    The MNIST set from `sklearn`.
    """
    def __init__(self):
        super().__init__(8)
        self.digits = load_digits()
        self._split_sets()

    def _split_sets(self):
        self.training_set, self.testing_set = train_test_split(self.digits.data)
        self.training_set, self.validating_set = train_test_split(self.training_set)

def main():
    mnist = MnistSet8()
    print(mnist)
    mnist.print_test_image(int(sys.argv[1]) if len(sys.argv) >= 2 else 10)

if __name__ == '__main__':
    main()
