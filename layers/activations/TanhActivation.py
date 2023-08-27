import numpy as np

from layers.activations import BaseActivation


class TanhActivation(BaseActivation):
    def function(self, x: np.array) -> np.array:
        return np.tanh(x)

    def function_prime(self, x: np.array) -> np.array:
        return 1 - np.tanh(x)**2
