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
        self.training_set, self.testing_set = train_test_split(load_digits().data)
        self.training_set, self.validating_set = train_test_split(self.training_set)

    def print_test_image(self, index: int):
        image = self.testing_set[index]
        for i in range(8):
            for j in range(8):
                color = max(0, min(int(16 * image[8 * i + j]), 255))

                print(f'\x1b[38;2;{color};{color};{color}m\u2588', end='')
            print('')
        print('\x1b[0m', end='')

def main():
    """
    Usage: `python mnist8.py [image-index]`
    """
    mnist = MnistSet8()
    print(mnist)
    mnist.print_test_image(int(sys.argv[1]) if len(sys.argv) >= 2 else 0)

if __name__ == '__main__':
    main()
