import keras

if keras.backend.backend() == "tensorflow":
    import tensorflow as tf

    from .tensorflow import complex, ema, segment_ops, static

    BackendTensor = tf.Tensor
elif keras.backend.backend() == "torch":
    import torch

    from .torch import complex, ema, segment_ops, static

    BackendTensor = torch.Tensor
elif keras.backend.backend() == "jax":
    import jax.numpy as jnp

    from .jax import complex, ema, segment_ops, static

    BackendTensor = jnp.ndarray
else:
    raise RuntimeError(f"Backend '{keras.backend.backend()}' not supported")

__all__ = [
    "BackendTensor",
    "complex",
    "ema",
    "segment_ops",
    "static",
]
