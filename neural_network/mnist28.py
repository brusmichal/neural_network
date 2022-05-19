# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
import sys

from sklearn.model_selection import train_test_split
from typing import List, Tuple

from mnist_base import MnistSetBase

@dataclass
class Mnist28Source:
    label_filename: str
    data_filename: str

class MnistSet28(MnistSetBase):
    """
    Original MNIST set.
    """
    def __init__(self, training_filename: str, testing_filename: str):
        super().__init__(28)
        self.training_set = self._read_set(training_filename)
        self.training_set, self.validating_set = train_test_split(self.training_set)
        self.testing_set = self._read_set(testing_filename)

    def _read_set(self, source: Mnist28Source) -> List[List[int]]:
        return [
            (self._adjust_pixels(pixels), label) for pixels, label
            in zip(self._read_pixels(source.data_filename), self._read_labels(source.label_filename))
        ]

    def _read_labels(self, filename: str) -> List[int]:
        with open(filename, 'rb') as file:
            assert int.from_bytes(file.read(2), byteorder='big') == 0
            assert int.from_bytes(file.read(1), byteorder='big') == 0x08
            assert int.from_bytes(file.read(1), byteorder='big') == 1
            label_count = int.from_bytes(file.read(4), byteorder='big')
            return [int.from_bytes(file.read(1), byteorder='big') for _ in range(label_count)]

    def _adjust_pixels(self, pixels):
        return [pixel / 255 for pixel in pixels]

    def _read_pixels(self, filename: str) -> List[List[int]]:
        with open(filename, 'rb') as file:
            image_count, image_size = self._read_pixels_header(file)
            return [file.read(image_size) for _ in range(image_count)]

    def _read_pixels_header(self, file) -> Tuple[int, int]:
        assert int.from_bytes(file.read(2), byteorder='big') == 0
        assert int.from_bytes(file.read(1), byteorder='big') == 0x08
        assert int.from_bytes(file.read(1), byteorder='big') == 3
        dims = [int.from_bytes(file.read(4), byteorder='big') for _ in range(3)]
        return dims[0], dims[1] * dims[2]

DEFAULT_TRAINING_FILES = Mnist28Source('train-labels.idx1-ubyte', 'train-images.idx3-ubyte')
DEFAULT_TESTING_FILES = Mnist28Source('t10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte')
DEFAULT_MNIST28 = MnistSet28(DEFAULT_TRAINING_FILES, DEFAULT_TESTING_FILES)

if __name__ == '__main__':
    """
    Usage: `python mnist28.py [image-index]`
    """
    mnist = DEFAULT_MNIST28
    print(mnist)
    mnist.print_test_image(int(sys.argv[1]) if len(sys.argv) >= 2 else 0)
