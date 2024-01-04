import jax
import jax.numpy as jnp
import numba
import numpy as np
from jax.core import ShapedArray

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


add_and_mul_jax = jambax.numba_to_jax(
    "add_and_mul", add_and_mul, add_and_mul_shape_fn, nopython=True
)
add_and_mul_jax_jit = jax.jit(add_and_mul_jax, device=jax.devices("cpu")[0])
x = np.random.uniform(size=(5,)).astype("float32")
y = np.random.uniform(size=(5,)).astype("float32")
z = np.random.uniform(size=(5,)).astype("float32")

results = add_and_mul_jax(x, y, z)
print(results)
results = add_and_mul_jax(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
print(results)
results = add_and_mul_jax_jit(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
print(results)
