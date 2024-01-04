import functools
import math
import typing as tp

import keras


def _calculate_scale(nonlinearity, param=None):
    """
    Return the recommended scale value for the given nonlinearity function.
    The values are as follows.

    This is equivalent to `torch.nn.init.calculate_gain(**kwargs)**2`."""
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 25.0 / 9
    elif nonlinearity == "relu":
        return 2.0
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return 2.0 / (1 + negative_slope**2)
    elif nonlinearity == "selu":
        return (
            9.0 / 16
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _kaiming(
    distribution: str,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> keras.initializers.Initializer:
    scale = _calculate_scale(nonlinearity, a)
    return keras.initializers.VarianceScaling(
        scale=scale, mode=mode, distribution=distribution
    )


def kaiming_normal(
    a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> keras.initializers.Initializer:
    return _kaiming("untruncated_normal", a=a, mode=mode, nonlinearity=nonlinearity)


def kaiming_uniform(
    a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> keras.initializers.Initializer:
    return _kaiming("uniform", a=a, mode=mode, nonlinearity=nonlinearity)


def default_linear_kernel_initializer() -> keras.initializers.Initializer:
    # return kaiming_uniform(math.sqrt(5.0))
    return keras.initializers.VarianceScaling(scale=1.0 / 3, distribution="uniform")


def default_linear_bias_initializer(fan_in: int) -> keras.initializers.Initializer:
    bound = 1.0 / math.sqrt(fan_in)
    return keras.initializers.RandomUniform(-bound, bound)


default_conv_kernel_initializer = default_linear_kernel_initializer


def default_conv_bias_initializer(
    filters_in: int, kernel_shape: tp.Tuple[int, ...]
) -> keras.initializers.Initializer:
    return default_linear_bias_initializer(
        functools.reduce(lambda a, b: a * b, kernel_shape, filters_in)
    )
