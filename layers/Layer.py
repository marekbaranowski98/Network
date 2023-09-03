from abc import abstractmethod


class Layer:
    def __init__(self):
        self.__input = None
        self.__output = None

    @abstractmethod
    def forward(self, x): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, learning_rate): # pragma: no cover
        raise NotImplementedError
