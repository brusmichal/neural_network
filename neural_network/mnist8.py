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
        digits = load_digits()
        images = [(self._adjust_pixels(pixels), label) for pixels, label in zip(digits.data, digits.target)]
        self.training_set, self.testing_set = train_test_split(images)
        self.training_set, self.validating_set = train_test_split(self.training_set)

    def _adjust_pixels(self, pixels):
        return [pixel / 16 for pixel in pixels]


if __name__ == '__main__':
    """
    Usage: `python mnist8.py [image-index]`
    """
    mnist = MnistSet8()
    print(mnist)
    mnist.print_test_image(int(sys.argv[1]) if len(sys.argv) >= 2 else 0)
