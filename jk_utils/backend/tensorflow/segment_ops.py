import functools
import string
import typing as tp

import jax
from jax import lax
from jax.experimental import jax2tf


def _wrapped_segment_op(jax_segment_op: tp.Callable):
    @functools.cache
    def get_tf_func(
        ndims: int,
        *,
        num_segments: tp.Optional[int] = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        bucket_size: tp.Optional[int] = None,
        mode: tp.Optional[lax.GatherScatterMode] = None,
    ):
        return jax2tf.convert(
            jax.jit(
                functools.partial(
                    jax_segment_op,
                    num_segments=num_segments,
                    indices_are_sorted=indices_are_sorted,
                    unique_indices=unique_indices,
                    bucket_size=bucket_size,
                    mode=mode,
                )
            ),
            polymorphic_shapes=[
                f"({', '.join(string.ascii_lowercase[:ndims])})",
                "(a,)",
            ],
        )

    def return_func(
        data,
        segment_ids,
        num_segments: tp.Optional[int] = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        bucket_size: tp.Optional[int] = None,
        mode: tp.Optional[lax.GatherScatterMode] = None,
    ):
        return get_tf_func(
            len(data.shape),
            num_segments=num_segments,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            bucket_size=bucket_size,
            mode=mode,
        )(data, segment_ids)

    functools.update_wrapper(return_func, jax_segment_op)
    return_func.__doc__ = (
        f"tensorflow wrapper around jax.ops.{jax_segment_op.__name__}\n\n"
        + jax_segment_op.__doc__
    )
    return return_func


segment_sum = _wrapped_segment_op(jax.ops.segment_sum)
segment_prod = _wrapped_segment_op(jax.ops.segment_prod)
segment_max = _wrapped_segment_op(jax.ops.segment_max)
segment_min = _wrapped_segment_op(jax.ops.segment_min)

__all__ = [
    "segment_sum",
    "segment_prod",
    "segment_max",
    "segment_min",
]
