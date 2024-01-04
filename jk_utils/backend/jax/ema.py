import jax.numpy as jnp
from jax_stsc_ops import cumulative_ema as cema

# def _ema_add(a, b):
#     x_a, f_a = a
#     x_b, f_b = b
#     x_out = x_a * f_b + x_b
#     f_out = f_a * f_b
#     return x_out, f_out


# def ema(x: jnp.ndarray, factors: jnp.ndarray, axis: int):
#     acc = jax.lax.associative_scan(_ema_add, (x, factors), axis=axis)
#     return acc[0]


# def _segment_ema_add(a, b):
#     xa, factor_a, sa = a
#     xb, factor_b, sb = b
#     is_same = sa == sb
#     x_out = jnp.where(is_same, xa * factor_b + xb, xb)
#     factor_out = jnp.where(is_same, factor_a * factor_b, factor_b)
#     return x_out, factor_out, sb


# def segment_ema(
#     x: jnp.ndarray, factors: jnp.ndarray, segment_ids: jnp.ndarray, axis: int
# ):
#     assert len(x.shape) == len(factors.shape) == len(segment_ids.shape), (
#         x.shape,
#         factors.shape,
#         segment_ids.shape,
#     )
#     x, factors, segment_ids = jnp.broadcast_arrays(x, factors, segment_ids)
#     acc = jax.lax.associative_scan(
#         _segment_ema_add, (x, factors, segment_ids), axis=axis
#     )
#     return acc[0]


# HACK


def segment_ema(
    x: jnp.ndarray, factors: jnp.ndarray, segment_ids: jnp.ndarray, axis: int
):
    assert axis == 0, axis
    assert all(s == 1 for s in segment_ids.shape[1:])
    segment_ids = jnp.squeeze(segment_ids)
    mask = jnp.pad(segment_ids[:-1] == segment_ids[1:], [[1, 0]])
    mask = mask.reshape(mask.shape[0], *(1 for _ in factors.shape[1:]))
    factors = factors * mask.astype(factors.dtype)
    x, factors = jnp.broadcast_arrays(x, factors)
    return ema(x, factors, axis=0)


def ema(x: jnp.ndarray, factors: jnp.ndarray, axis: int):
    assert axis == 0, axis
    n = x.shape[0]
    out = (
        cema.cumulative_ema(
            x.reshape(n, -1).T.reshape(-1), factors.reshape(n, -1).T.reshape(-1)
        )
        .reshape(-1, n)
        .T
    )
    return out.reshape(x.shape)


# def segment_ema(
#     x: jnp.ndarray, factors: jnp.ndarray, segment_ids: jnp.ndarray, axis: int
# ):
#     if len(x.shape) == 1:
#         assert len(factors.shape) == 1
#         assert len(segment_ids.shape) == 1
#         return cema.segment_cumulative_ema(x, factors, segment_ids)
#     assert axis == 0, axis
#     if len(x.shape) == 3:
#         x, factors = jnp.broadcast_arrays(x, factors)
#         x_shape = x.shape
#         size = x_shape[0]
#         assert all(s == 1 for s in segment_ids.shape[1:]), segment_ids.shape[1:]
#         x = x.reshape(size, -1)
#         out = _tiled_segment_ema(
#             x.reshape(size, -1), factors.reshape(size, -1), segment_ids
#         )
#         return out.reshape(x_shape)
#     return _tiled_segment_ema(x, factors, segment_ids)


# def _tiled_segment_ema(x: jnp.ndarray, factors: jnp.ndarray, segment_ids: jnp.ndarray):
#     assert len(x.shape) == 2, (x.shape, factors.shape, segment_ids.shape)
#     assert x.shape == factors.shape
#     assert segment_ids.shape[0] == x.shape[0]
#     assert segment_ids.shape[1] == 1

#     size, num_channels = x.shape
#     segment_ids = segment_ids * num_channels + jnp.arange(num_channels)
#     return (
#         cema.segment_cumulative_ema(
#             x.T.flatten(), factors.T.flatten(), segment_ids.T.flatten()
#         )
#         .reshape(num_channels, size)
#         .T
#     )
