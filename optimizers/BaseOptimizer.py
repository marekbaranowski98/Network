class BaseOptimizer:
    def function(self, y_true, y_pred):
        raise NotImplementedError

    def function_prime(self, y_true, y_pred):
        raise NotImplementedError
