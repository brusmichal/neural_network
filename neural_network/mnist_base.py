# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
from typing import List

from pyrsistent import freeze

@dataclass
class MnistImage:
    label: int
    pixels: List[int]

    def get_label(self) -> int:
        assert 0 <= self.label and self.label <= 9
        return self.label

class MnistSetBase:
    """
    The base for 28x28 and 8x8 MNIST sets.
    """
    def __init__(self):
        self.training_set: List[MnistImage] = []
        self.validating_set: List[MnistImage] = []
        self.testing_set: List[MnistImage] = []

    def get_training_set(self) -> List[MnistImage]:
        return self.training_set

    def get_validating_set(self) -> List[MnistImage]:
        return self.validating_set

    def get_testing_set(self) -> List[MnistImage]:
        return self.training_set

    def get_training_images(self) -> List[List[int]]:
        return [img.pixels for img in self.training_set]

    def get_training_labels(self) -> List[int]:
        return [img.get_label() for img in self.training_set]

    def get_validating_images(self) -> List[List[int]]:
        return [img.pixels for img in self.validating_set]

    def get_validating_labels(self) -> List[int]:
        return [img.get_label() for img in self.validating_set]

    def get_testing_images(self) -> List[List[int]]:
        return [img.pixels for img in self.testing_set]

    def get_testing_labels(self) -> List[int]:
        return [img.get_label() for img in self.testing_set]

    def __repr__(self) -> str:
        return '\n'.join([
            f'Training set size:   {len(self.training_set)}',
            f'Validating set size: {len(self.validating_set)}',
            f'Testing set size:    {len(self.testing_set)}'
        ])
