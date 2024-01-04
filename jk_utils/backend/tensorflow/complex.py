import tensorflow as tf


def complex(real, imag):
    return tf.complex(real, imag)


def is_complex(x) -> bool:
    return "complex" in str(x.dtype)


def exp(x):
    return tf.exp(x)
