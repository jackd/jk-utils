import typing as tp
import unittest

import keras
import torch
from absl.testing import parameterized

from jk_utils import testing
from jk_utils.initializers import torch_compat


class TorchCompatInitializerTest(testing.TestCase, parameterized.TestCase):
    @parameterized.product(
        mode=("fan_in", "fan_out"),
        nonlinearity=("linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu"),
        a=(None, 1.5),
    )
    def test_kaiming_uniform(
        self,
        kernel_shape: tp.Tuple[int, ...] = (13,),
        channels_in: int = 64,
        channels_out: int = 128,
        mode="fan_in",
        nonlinearity="relu",
        a=None,
    ):
        torch_shape = (channels_out, channels_in, *kernel_shape)
        keras_shape = (*kernel_shape, channels_in, channels_out)
        actual = torch_compat.kaiming_uniform(
            mode=mode, nonlinearity=nonlinearity, a=a
        )(keras_shape)
        actual = keras.ops.convert_to_numpy(actual).flatten()

        expected = torch.zeros(torch_shape)
        torch.nn.init.kaiming_uniform_(
            expected, a=a, mode=mode, nonlinearity=nonlinearity
        )
        expected = expected.numpy()

        self.assertAllClose(actual.mean(), expected.mean(), atol=1e-2)
        self.assertAllClose(actual.std(), expected.std(), atol=1e-3)

    @parameterized.product(
        mode=("fan_in", "fan_out"),
        nonlinearity=("linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu"),
        a=(None, 1.5),
    )
    def test_kaiming_normal(
        self,
        kernel_shape: tp.Tuple[int, ...] = (13,),
        channels_in: int = 64,
        channels_out: int = 128,
        mode="fan_in",
        nonlinearity="relu",
        a=None,
    ):
        torch_shape = (channels_out, channels_in, *kernel_shape)
        keras_shape = (*kernel_shape, channels_in, channels_out)
        actual = torch_compat.kaiming_normal(mode=mode, nonlinearity=nonlinearity, a=a)(
            keras_shape
        )
        actual = keras.ops.convert_to_numpy(actual).flatten()

        expected = torch.zeros(torch_shape)
        torch.nn.init.kaiming_normal_(
            expected, a=a, mode=mode, nonlinearity=nonlinearity
        )
        expected = expected.numpy()

        self.assertAllClose(actual.mean(), expected.mean(), atol=1e-2)
        self.assertAllClose(actual.std(), expected.std(), atol=1e-3)

    def test_default_linear_kernel_initializer(
        self,
        channels_in: int = 64,
        channels_out: int = 128,
    ):
        keras_shape = (channels_in, channels_out)
        actual = torch_compat.default_linear_kernel_initializer()(keras_shape)
        actual = keras.ops.convert_to_numpy(actual).flatten()

        l = torch.nn.Linear(channels_in, channels_out)
        expected = l.weight.detach().numpy().flatten()

        self.assertAllClose(actual.mean(), expected.mean(), atol=1e-2)
        self.assertAllClose(actual.std(), expected.std(), atol=1e-3)

    def test_default_linear_bias_initializer(
        self,
        channels_in: int = 512,
        channels_out: int = 1024,
    ):
        actual = torch_compat.default_linear_bias_initializer(channels_in)(
            (channels_out,)
        )
        actual = keras.ops.convert_to_numpy(actual).flatten()

        l = torch.nn.Linear(channels_in, channels_out)
        expected = l.bias.detach().numpy().flatten()

        self.assertAllClose(actual.mean(), expected.mean(), atol=1e-2)
        self.assertAllClose(actual.std(), expected.std(), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
