class BaseActivation:
    def function(self, x):
        raise NotImplementedError

    def function_prime(self, x):
        raise NotImplementedError
