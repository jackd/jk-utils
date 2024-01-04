import jax
import numba
import numpy as np
import tensorflow as tf
from jax.core import ShapedArray
from jax.experimental import jax2tf

from jk_utils.numba_utils import jambax


@numba.njit()
def add_and_mul(args):
    output_1, output_2, input_1, input_2, input_3 = args
    # Now edit output_1 and output_2 *in place*.
    output_1.fill(0)
    output_2.fill(0)
    output_1 += input_1 + input_2
    output_2 += input_1 * input_3


def add_and_mul_shape_fn(input_1, input_2, input_3):
    assert input_1.shape == input_2.shape
    assert input_1.shape == input_3.shape
    return (
        ShapedArray(input_1.shape, input_1.dtype),
        ShapedArray(input_1.shape, input_1.dtype),
    )


add_and_mul_jax = jambax.numba_to_jax("add_and_mul", add_and_mul, add_and_mul_shape_fn)
add_and_mul_jax_jit = jax.jit(add_and_mul_jax, device=jax.devices("cpu")[0])
add_and_mul_tf = jax2tf.convert(
    add_and_mul_jax_jit,
    native_serialization_platforms=("cpu",),
    native_serialization=False,
)

x = np.random.uniform(size=(5,)).astype("float32")
y = np.random.uniform(size=(5,)).astype("float32")
z = np.random.uniform(size=(5,)).astype("float32")

with tf.device("/cpu:0"):
    add_and_mul_tf_jit = tf.function(add_and_mul_tf, jit_compile=True)
    # results = add_and_mul_tf(x, y, z)
    # print(results)
    # results = add_and_mul_tf(
    #     tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)
    # )
    # print(results)
    results = add_and_mul_tf_jit(
        tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(z)
    )
    print(results)
