import unittest
import numpy as np

from layers import Linear


class TestLinearBackward(unittest.TestCase):
    def __generate_linear(self, input_shape: int, output_shape: int, input_data: np.array) -> Linear:
        layer: Linear = Linear(input_shape, output_shape)
        layer.forward(input_data)

        return layer

    def test_backward_shape(self):
        input_shape = 5
        output_shape = 3
        batch_size = 1
        input_data = np.random.rand(batch_size, input_shape)
        layer: Linear = self.__generate_linear(input_shape, output_shape, input_data)

        output = layer.backward(np.random.rand(batch_size, output_shape), 1e-3)

        self.assertEqual(
            output.shape,
            (batch_size, input_shape),
            'Expected output shape is another to actual.'
        )

    def test_backward_calculate(self):
        input_shape = 5
        output_shape = 3
        batch_size = 1
        learning_rate = 1e-3
        input_data = np.random.rand(batch_size, input_shape)
        layer: Linear = self.__generate_linear(input_shape, output_shape, input_data)

        gradient = np.random.rand(batch_size, output_shape)
        expected_input_error = np.dot(gradient, layer.weight.T)
        expected_weight = layer.weight - learning_rate * np.dot(input_data.T, gradient)
        expected_bias = layer.bias - learning_rate * gradient

        output_backward = layer.backward(gradient, learning_rate)

        np.testing.assert_allclose(
            output_backward,
            expected_input_error,
            rtol=1e-5,
            err_msg='Expected input error is another to actual.'
        )

        np.testing.assert_allclose(
            layer.weight,
            expected_weight,
            rtol=1e-5,
            err_msg='Expected weight error is another to actual.'
        )

        np.testing.assert_allclose(
            layer.bias,
            expected_bias,
            rtol=1e-5,
            err_msg='Expected bias error is another to actual.'
        )

    def test_backward_manual_calculate(self):
        input_shape = 2
        output_shape = 2
        learning_rate = 1e-3
        input_data = np.array([[1.0, 2.0]])

        layer: Linear = self.__generate_linear(input_shape, output_shape, input_data)
        layer.weight = np.array([[0.3, 0.6], [0.8, 0.7]])
        layer.bias = np.array([[0.4, 0.5]])

        gradient = np.array([[1, 1]])
        expected_input_error = np.array([[0.9, 1.5]])
        expected_weight = np.array([[0.299, 0.599], [0.798, 0.698]])
        expected_bias = np.array([[0.399, 0.499]])

        output_backward = layer.backward(gradient, learning_rate)

        np.testing.assert_allclose(
            output_backward,
            expected_input_error,
            rtol=1e-5,
            err_msg='Expected input error is another to actual.'
        )

        np.testing.assert_allclose(
            layer.weight,
            expected_weight,
            rtol=1e-5,
            err_msg='Expected weight error is another to actual.'
        )

        np.testing.assert_allclose(
            layer.bias,
            expected_bias,
            rtol=1e-5,
            err_msg='Expected bias error is another to actual.'
        )
