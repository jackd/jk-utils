import functools
import typing as tp

import jax
import jax.numpy as jnp


def _static_bincount(x, weights=None, length=None, axis: int = 0):
    # base recursion case
    if len(x.shape) == 1:
        assert axis == 0, axis
        return jnp.bincount(x, weights, length=length)
    args = (x,) if weights is None else (x, weights)
    if axis == 0:
        return jax.vmap(
            functools.partial(_static_bincount, length=length, axis=axis),
            in_axes=1,
            out_axes=1,
        )(*args)
    # axis > 0
    return jax.vmap(
        functools.partial(_static_bincount, length=length, axis=axis - 1),
        in_axes=0,
        out_axes=0,
    )(*args)


def static_bincount(
    x: jnp.ndarray,
    weights: tp.Optional[jnp.ndarray] = None,
    length: tp.Optional[int] = None,
    axis: int = 0,
) -> jnp.ndarray:
    if weights is not None:
        x, weights = jnp.broadcast_arrays(x, weights)

    if axis < 0:
        axis += len(x.shape)
    assert 0 <= axis < len(x.shape), (axis, x.shape)
    return _static_bincount(x, weights, length, axis)


def set_shape(x, shape):
    assert x.shape == shape, (x.shape, shape)
    return x
