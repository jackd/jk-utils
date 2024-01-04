import tensorflow as tf
import tensorflow_probability as tfp


def _ema_add(a, b):
    x_a, f_a = a
    x_b, f_b = b
    x_out = x_a * f_b + x_b
    f_out = f_a * f_b
    return x_out, f_out


def ema(x: tf.Tensor, factors: tf.Tensor, axis: int):
    acc = tfp.math.scan_associative(_ema_add, (x, factors), axis=axis)
    return acc[0]


def _segment_ema_add(a, b):
    xa, factor_a, sa = a
    xb, factor_b, sb = b
    is_same = sa == sb
    x_out = tf.where(is_same, xa * factor_b + xb, xb)
    factor_out = tf.where(is_same, factor_a * factor_b, factor_b)
    return x_out, factor_out, sb


def segment_ema(x: tf.Tensor, factors: tf.Tensor, segment_ids: tf.Tensor, axis: int):
    assert len(x.shape) == len(factors.shape) == len(segment_ids.shape), (
        x.shape,
        factors.shape,
        segment_ids.shape,
    )
    acc = tfp.math.scan_associative(
        _segment_ema_add, (x, factors, segment_ids), axis=axis
    )
    return acc[0]


# import functools
# import string
# import tensorflow as tf
# import jax
# from jax.experimental import jax2tf
# from ..jax import ema as _jax_ema


# def _polymorphic_shape(shape):
#     shape = [
#         string.ascii_letters[i] if s is None else str(s) for i, s in enumerate(shape)
#     ]
#     return f"({', '.join(shape)})"


# @functools.cache
# def _ema_func(axis: int, *shapes):
#     return jax2tf.convert(
#         jax.jit(functools.partial(_jax_ema.ema, axis=axis)),
#         polymorphic_shapes=tuple(_polymorphic_shape(s) for s in shapes),
#     )


# def ema(x: tf.Tensor, factors: tf.Tensor, axis: int) -> tf.Tensor:
#     return _ema_func(axis, x.shape, factors.shape)(x, factors)


# @functools.cache
# def _segment_ema_func(axis: int, *shapes):
#     return jax2tf.convert(
#         jax.jit(functools.partial(_jax_ema.segment_ema, axis=axis)),
#         polymorphic_shapes=tuple(_polymorphic_shape(s) for s in shapes),
#     )


# def segment_ema(
#     x: tf.Tensor, factors: tf.Tensor, segment_ids: tf.Tensor, axis: int
# ) -> tf.Tensor:
#     return _segment_ema_func(axis, x.shape, factors.shape, segment_ids.shape)(
#         x, factors, segment_ids
#     )
