# Author: Jakub Mazurkiewicz

class MnistSetBase:
    def __init__(self):
        self.training_set = []
        self.validating_set = []
        self.testing_set = []

    def get_training_set(self):
        return self.training_set

    def get_validating_set(self):
        return self.validating_set

    def get_testing_set(self):
        return self.training_set

    def __repr__(self):
        return '\n'.join([
            f'Training set size:   {len(self.training_set)}',
            f'Validating set size: {len(self.validating_set)}',
            f'Testing set size:    {len(self.testing_set)}'
        ])
