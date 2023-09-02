import numpy as np

from optimizers import Optimizer


class MSE(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def function(self, y_true: np.array, y_pred: np.array) -> np.array:
        return np.mean(
            np.power(y_true - y_pred, 2)
        )

    def function_prime(self, y_true: np.array, y_pred: np.array) -> np.array:
        return 2 * (y_pred - y_true) / y_true.size
