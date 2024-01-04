import jax.numpy as jnp


def complex(real, imag):
    return real + 1j * imag


def is_complex(x) -> bool:
    return "complex" in str(x.dtype)


def exp(x):
    return jnp.exp(x)
