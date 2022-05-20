# Author: Jakub Mazurkiewicz

class MnistSetBase:
    """
    The base for 28x28 and 8x8 MNIST sets.
    """
    def __init__(self, size):
        self.size = size
        self.training_set = []
        self.validating_set = []
        self.testing_set = []

    def get_training_set(self):
        return self.training_set

    def get_validating_set(self):
        return self.validating_set

    def get_testing_set(self):
        return self.training_set

    def get_training_images(self):
        return [img[0] for img in self.training_set]

    def get_training_labels(self):
        return [img[1] for img in self.training_set]

    def get_validating_images(self):
        return [img[0] for img in self.validating_set]

    def get_validating_labels(self):
        return [img[1] for img in self.validating_set]

    def get_testing_images(self):
        return [img[0] for img in self.testing_set]

    def get_testing_labels(self):
        return [img[1] for img in self.testing_set]

    def __repr__(self) -> str:
        return '\n'.join([
            f'Training set size:   {len(self.training_set)}',
            f'Validating set size: {len(self.validating_set)}',
            f'Testing set size:    {len(self.testing_set)}'
        ])

    def print_test_image(self, index: int):
        image = self.get_testing_images()[index]
        for i in range(self.size):
            for j in range(self.size):
                color = int(255 * image[self.size * i + j])
                print(f'\x1b[38;2;{color};{color};{color}m\u2588', end='')
            print('')
        print(f'\x1b[0mLABEL: {self.get_testing_labels()[index]}')
