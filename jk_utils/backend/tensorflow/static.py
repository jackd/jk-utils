import typing as tp

import tensorflow as tf

tfnp = tf.experimental.numpy


def _static_bincount(
    x: tf.Tensor,
    weights: tp.Optional[tf.Tensor] = None,
    length: tp.Optional[int] = None,
) -> tf.Tensor:
    x.shape.assert_has_rank(1)
    if weights is None:
        y = tf.math.bincount(x, minlength=length, maxlength=length)
    else:
        y = tf.math.unsorted_segment_sum(weights, x, num_segments=length)
    if length is not None:
        y.set_shape((length,))
    return y


def static_bincount(
    x: tf.Tensor,
    weights: tp.Optional[tf.Tensor] = None,
    length: tp.Optional[int] = None,
    axis: int = 0,
):
    x = tf.convert_to_tensor(x)
    if weights is not None:
        weights = tf.broadcast_to(weights, x.shape)
    if axis < 0:
        axis += len(x.shape)
    assert 0 <= axis < len(x.shape), (axis, x.shape)
    if len(x.shape) == 1:
        assert axis == 0, axis
        return _static_bincount(x, weights, length=length)
    assert length is not None

    if axis != len(x.shape) - 1:
        x = tfnp.moveaxis(x, axis, -1)
        if weights is not None:
            weights = tfnp.moveaxis(weights, axis, -1)
    x_shape = tf.shape(x)
    d = x_shape[-1]
    x = tf.reshape(x, (-1, d))
    if weights is not None:
        weights = tf.reshape(weights, (-1, d))

    args = (x,) if weights is None else (x, weights)
    out = tf.vectorized_map(lambda args: _static_bincount(*args, length=length), args)
    out = tf.reshape(out, (*tf.unstack(x_shape[:-1]), length))
    if axis != len(x.shape) - 1:
        out = tfnp.moveaxis(out, -1, axis)
    return out


def set_shape(x, shape):
    x.set_shape(shape)
    return x
