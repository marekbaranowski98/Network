import numpy as np

from tqdm import tqdm

from layers import BaseLayer
from optimizers import BaseOptimizer


class Network:
    def __init__(self):
        self.__layers = []
        self.__optimizer = None
        self.__metric = None
        self.__zero_grad()

    def add(self, layer: BaseLayer):
        self.__layers.append(layer)

    def configuration(self, optimizer: BaseOptimizer):
        self.__optimizer = optimizer

    def fit(
            self,
            x_train: np.array,
            y_train: np.array,
            epochs: int,
            x_val: np.array = None,
            y_val: np.array = None,
            batch: int = 1,
            verbose: bool = False
    ):
        if self.__optimizer is None:
            raise Exception('Add network configurations before running a fit.')

        for epoch in range(epochs):
            self.__zero_grad()
            with tqdm(
                total=len(x_train) + (len(y_val) if y_val else 0),
                unit='batch'
            ) as tepoch:
                tepoch.set_description(f'Epoch {epoch+1}/{epochs}')

                for sample_idx in range(len(x_train)):
                    self.__train(x_train[sample_idx], y_train[sample_idx])
                    tepoch.update()

                    if x_val and y_val:
                        self.__valid()
                        tepoch.update()

    def predict(self, x):
        return self.__forward(x)

    def summary(self):
        pass

    def __train(self, x: np.array, y: np.array) -> np.array:
        out = self.__forward(x)
        self.__error += self.__optimizer.function(y, out)

        self.__backward(y, out)

    def __valid(self):
        pass

    def __forward(self, x: np.array) -> np.array:
        output = x

        for layer in self.__layers:
            output = layer.forward(output)

        return output

    def __backward(self, y: np.array, pred: np.array):
        error = self.__optimizer.function_prime(np.array(y), pred)
        for layer in reversed(self.__layers):
            error = layer.backward(error, self.__optimizer.get_learning_rate())

    def __zero_grad(self):
        self.__error = 0
