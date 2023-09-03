class Optimizer:
    def __init__(self, learning_rate):
        self.lr: float = learning_rate

    def function(self, y_true, y_pred):
        raise NotImplementedError

    def function_prime(self, y_true, y_pred):
        raise NotImplementedError
