import numpy as np

from layers import Layer


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int, batch_size: int = 1):
        super().__init__()
        self.weight: np.array = np.random.rand(input_size, output_size)
        self.bias: np.array = np.random.rand(batch_size, output_size)

    def forward(self, x: np.array) -> np.array:
        self.__input = x
        self.__output = np.dot(self.__input, self.weight) + self.bias
        return self.__output

    def backward(self, output_error: np.array, learning_rate: float) -> np.array:
        input_error = np.dot(output_error, self.weight.T)
        weight_error = np.dot(self.__input.T, output_error)

        self.weight -= learning_rate * weight_error
        self.bias -= learning_rate * output_error

        return input_error
