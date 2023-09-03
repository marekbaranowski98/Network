import unittest
import numpy as np

from layers import Linear


class TestLinearForward(unittest.TestCase):
    layer: Linear = None

    def __generate_output_linear(self, input_shape: int, output_shape: int, input_data: np.array) -> np.array:
        self.layer = Linear(input_shape, output_shape)
        return self.layer.forward(input_data)

    def test_forward_output_shape_single_element(self):
        input_shape = 5
        output_shape = 3
        batch_size = 1
        input_data = np.random.rand(batch_size, input_shape)

        output = self.__generate_output_linear(input_shape, output_shape, input_data)

        self.assertEqual(
            output.shape,
            (batch_size, output_shape),
            'Expected output shape is another to actual.'
        )

    def test_forward_output_shape_batch(self):
        input_shape = 5
        output_shape = 3
        batch_size = 16
        input_data = np.random.rand(batch_size, input_shape)

        output = self.__generate_output_linear(input_shape, output_shape, input_data)

        self.assertEqual(
            output.shape,
            (batch_size, output_shape),
            'Expected output shape is another to actual.'
        )

    def test_forward_output(self):
        input_shape = 5
        output_shape = 3
        batch_size = 1
        input_data = np.random.rand(batch_size, input_shape)

        output = self.__generate_output_linear(input_shape, output_shape, input_data)

        expected_output = np.dot(input_data, self.layer.weight) + self.layer.bias

        np.testing.assert_allclose(
            output,
            expected_output,
            rtol=1e-5,
            err_msg='Expected output is different from the one received.'
        )

    def test_forward_manual_calculation(self):
        input_shape = 2
        output_shape = 2
        batch_size = 1
        linear = Linear(input_shape, output_shape, batch_size)

        input_data = np.array([[1.0, 2.0]])
        linear.weight = np.array([[0.3, 0.6], [0.8, 0.7]])
        linear.bias = np.array([[0.4, 0.5]])

        output = linear.forward(input_data)
        expected_output = np.array([[2.3, 2.5]])

        np.testing.assert_allclose(
            output,
            expected_output,
            rtol=1e-5,
            err_msg='Expected output is different from the one received.'
        )

