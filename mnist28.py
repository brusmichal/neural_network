# Author: Jakub Mazurkiewicz
import sys

from sklearn.model_selection import train_test_split
from typing import List, Tuple

from mnist_base import MnistSetBase

class MnistSet28(MnistSetBase):
    """
    Original MNIST set.
    """
    def __init__(self, training_filename: str, testing_filename: str):
        super().__init__(28)
        self.training_set = self._read_set(training_filename)
        self.training_set, self.validating_set = train_test_split(self.training_set)
        self.testing_set = self._read_set(testing_filename)

    def _read_set(self, filename) -> List[List[int]]:
        with open(filename, 'rb') as file:
            image_count, image_size = self._read_header(file)
            return [file.read(image_size) for _ in range(image_count)]

    def _read_header(self, file) -> Tuple[int, int]:
        assert int.from_bytes(file.read(2), byteorder='big') == 0
        assert int.from_bytes(file.read(1), byteorder='big') == 0x08
        assert int.from_bytes(file.read(1), byteorder='big') == 3
        dims = [int.from_bytes(file.read(4), byteorder='big') for _ in range(3)]
        return dims[0], dims[1] * dims[2]

def main():
    mnist = MnistSet28('train-images.idx3-ubyte', 't10k-images.idx3-ubyte')
    print(mnist)
    mnist.print_test_image(int(sys.argv[1]) if len(sys.argv) >= 2 else 0)

if __name__ == '__main__':
    main()
