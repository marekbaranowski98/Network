class BaseOptimizer:
    def __init__(self, learning_rate):
        self.__lr = learning_rate

    def update_learning_rate(self, learning_rate):
        self.__lr = learning_rate

    def get_learning_rate(self):
        return self.__lr

    def function(self, y_true, y_pred):
        raise NotImplementedError

    def function_prime(self, y_true, y_pred):
        raise NotImplementedError
