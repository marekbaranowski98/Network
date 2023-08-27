import numpy as np

from optimizers import BaseOptimizer


class MSELoss(BaseOptimizer):
    def function_prime(self, y_true: np.array, y_pred: np.array) -> np.array:
        return np.mean(np.power(y_true, y_pred, 2))

    def function_prime(self, y_true: np.array, y_pred: np.array) -> np.array:
        return 2 * (y_pred - y_true) / y_true.size
