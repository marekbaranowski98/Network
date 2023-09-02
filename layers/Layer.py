class Layer:
    def __init__(self):
        self.__input = None
        self.__output = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError
