# Author: Jakub Mazurkiewicz
from typing import List
class MnistSetBase:
    def __init__(self):
        self.training_set = []
        self.validating_set = []
        self.testing_set = []

    def get_training_set(self) -> List[List[int]]:
        return self.training_set

    def get_validating_set(self) -> List[List[int]]:
        return self.validating_set

    def get_testing_set(self) -> List[List[int]]:
        return self.training_set

    def __repr__(self) -> str:
        return '\n'.join([
            f'Training set size:   {len(self.training_set)}',
            f'Validating set size: {len(self.validating_set)}',
            f'Testing set size:    {len(self.testing_set)}'
        ])

    def print_test_image(self, index: int):
        raise NotImplemented('Method `print_test_image` shall be implemented by derived class')
