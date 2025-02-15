"""
Shamelessly lifted from https://github.com/josipd/jax/blob/master/jax/experimental/jambax.py

See also discussion at https://github.com/google/jax/issues/1870

Call Numba from jitted JAX functions.

# The interface

To call your Numba function from JAX, you have to implement:

  1. A Numba function following our calling convention.
  2. A function for abstractly evaluating the function, i.e., for specifying
     the output shapes and dtypes from the input ones.

## 1. The Numba function

The Numba function has to accept a *single* tuple argument and do not return
anythin, i.e. have type `Callable[tuple[numba.carray], None]`. The output and
input arguments are stored consecutively in the tuple. For example, if you want
to implement a function that takes three arrays and returns two, the Numba
function should look like:

```py
@numba.jit
def add_and_mul(args):
  output_1, output_2, input_1, input_2, input_3 = args
  # Now edit output_1 and output_2 *in place*.
  output_1.fill(0)
  output_2.fill(0)
  output_1 += input_1 + input_2
  output_2 += input_1 * input_3
```

Note that the output arguments have to be modified *in-place*. These arrays are
allocated and owned by XLA.

## 2. The abstract evaluation function

You also have to implement a function that tells JAX how to compute the shapes
and types of the outputs from the inputs.For more information, please refer to

https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#Abstract-evaluation-rules

For example, for the above function, the corresponding abstract eval function is

```py
def add_and_mul_shape_fn(input_1, input_2, input_3):
  assert input_1.shape == input_2.shape
  assert input_1.shape == input_3.shape
  return (jax.core.ShapedArray(input_1.shape, input_1.dtype),
          jax.core.ShapedArray(input_1.shape, input_1.dtype))
```

# Conversion

Now, what is left is to convert the function:

```py
add_and_mul_jax = jax.experimental.jambax.numba_to_jax(
    "add_and_mul", add_and_mul, add_and_mul_shape_fn)
```

You can JIT compile the function as
```py
add_and_mul_jit = jax.jit(add_and_mul_jax)
```

# Optional
## Derivatives

You can define a gradient for your function as if you were definining a custom
gradient for any other JAX function. You can follow the tutorial at:

https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html

## Batching / vmap

Batching along the first axes is implemented via jax.lax.map. To implement your
own bathing rule, see the documentation of `numba_to_jax`.
"""

import collections
import ctypes
from functools import partial  # pylint:disable=g-importing-member

import jax
import numba
import numba.typed as nb_typed
import numpy as onp
from jax.interpreters import batching, mlir, xla
from jaxlib import hlo_helpers, xla_client, xla_extension
from numba import types as nb_types


def _xla_shape_to_abstract(xla_shape):
    return jax.core.ShapedArray(xla_shape.dimensions(), xla_shape.element_type())


def _create_xla_target_capsule(ptr):
    xla_capsule_magic = b"xla._CUSTOM_CALL_TARGET"
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    return ctypes.pythonapi.PyCapsule_New(ptr, xla_capsule_magic, None)


def _np_evaluation_rule(call_fn, abstract_eval_fn, *args, **kwargs):
    output_shapes = abstract_eval_fn(*args)
    outputs = tuple(
        onp.empty(shape.shape, dtype=shape.dtype) for shape in output_shapes
    )
    inputs = tuple(onp.asarray(arg) for arg in args)
    call_fn(outputs + inputs, **kwargs)
    return outputs


def _naive_batching(call_fn, args, batch_axes):
    # TODO(josipd): Check that the axes are all zeros. Add support when only a
    #               subset of the arguments have to be batched.
    # TODO(josipd): Do this smarter than n CustomCalls.
    return tuple(jax.lax.map(lambda x: call_fn(*x), args)), batch_axes


def _xla_translation(numba_fn, abstract_eval_fn, xla_builder, *args):
    """Returns the XLA CustomCall for the given numba function.

    Args:
      numba_fn: A numba function. For its signature, see the module docstring.
      abstract_eval_fn: The abstract shape evaluation function.
      xla_builder: The XlaBuilder instance.
      *args: The positional arguments to be passed to `numba_fn`.
    Returns:
      The XLA CustomCall operation calling into the numba function.
    """
    input_shapes = [xla_builder.get_shape(arg) for arg in args]
    # TODO(josipd): Check that the input layout is the numpy default.
    output_abstract_arrays = abstract_eval_fn(
        *[_xla_shape_to_abstract(shape) for shape in input_shapes]
    )
    output_shapes = tuple(array.shape for array in output_abstract_arrays)
    output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
    layout_for_shape = lambda shape: range(len(shape) - 1, -1, -1)
    output_layouts = map(layout_for_shape, output_shapes)
    xla_output_shapes = [
        xla_client.Shape.array_shape(*arg)
        for arg in zip(output_dtypes, output_shapes, output_layouts)
    ]
    xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)

    input_dtypes = tuple(shape.element_type() for shape in input_shapes)
    input_dimensions = tuple(shape.dimensions() for shape in input_shapes)

    xla_call_sig = nb_types.void(
        nb_types.CPointer(nb_types.voidptr), nb_types.CPointer(nb_types.voidptr)
    )

    @numba.cfunc(xla_call_sig)
    def xla_custom_call_target(output_ptrs, input_ptrs):
        args = nb_typed.List()
        for i in range(len(output_shapes)):
            args.append(
                numba.carray(output_ptrs[i], output_shapes[i], dtype=output_dtypes[i])
            )
        for i in range(len(input_dimensions)):
            args.append(
                numba.carray(input_ptrs[i], input_dimensions[i], dtype=input_dtypes[i])
            )
        numba_fn(args)

    target_name = xla_custom_call_target.native_name.encode("ascii")
    capsule = _create_xla_target_capsule(xla_custom_call_target.address)
    xla_extension.register_custom_call_target(target_name, capsule, "Host")
    return xla_client.ops.CustomCallWithLayout(
        xla_builder,
        target_name,
        operands=args,
        shape_with_layout=xla_output_shape,
        operand_shapes_with_layout=input_shapes,
    )


def _mlir_lowering(numba_fn, ctx: mlir.LoweringRuleContext, *args, jit_kwargs=None):
    input_shapes = tuple(array.shape for array in ctx.avals_in)
    input_dtypes = tuple(array.dtype for array in ctx.avals_in)
    output_shapes = tuple(array.shape for array in ctx.avals_out)
    output_dtypes = tuple(array.dtype for array in ctx.avals_out)

    xla_call_sig = nb_types.void(
        nb_types.CPointer(nb_types.voidptr), nb_types.CPointer(nb_types.voidptr)
    )

    @numba.cfunc(xla_call_sig, **(jit_kwargs or {}))
    def xla_custom_call_target(output_ptrs, input_ptrs):
        args = nb_typed.List()
        for i in range(len(output_shapes)):
            args.append(
                numba.carray(output_ptrs[i], output_shapes[i], dtype=output_dtypes[i])
            )
        for i in range(len(input_shapes)):
            args.append(
                numba.carray(input_ptrs[i], input_shapes[i], dtype=input_dtypes[i])
            )
        numba_fn(args)

    target_name = xla_custom_call_target.native_name.encode("ascii")
    capsule = _create_xla_target_capsule(xla_custom_call_target.address)
    xla_client.register_custom_call_target(
        target_name,
        capsule,
        ctx.module_context.platform,
    )
    out_types = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    return hlo_helpers.custom_call(
        call_target_name=target_name,
        out_types=out_types,
        operands=args,
    )


def numba_to_jax(
    name: str,
    numba_fn,
    abstract_eval_fn,
    batching_fn=None,
    nopython: bool = True,
):
    """Create a jittable JAX function for the given Numba function.

    Args:
      name: The name under which the primitive will be registered.
      numba_fn: The function that can be compiled with Numba.
      abstract_eval_fn: The abstract evaluation function.
      batching_fn: If set, this function will be used when vmap-ing the returned
        function.
    Returns:
      A jitable JAX function.
    """
    primitive = jax.core.Primitive(name)
    primitive.multiple_results = True

    def abstract_eval_fn_always(*args, **kwargs):
        # Special-casing when only a single tensor is returned.
        shapes = abstract_eval_fn(*args, **kwargs)
        if not isinstance(shapes, collections.abc.Collection):
            return [shapes]
        else:
            return shapes

    primitive.def_abstract_eval(abstract_eval_fn_always)
    primitive.def_impl(partial(_np_evaluation_rule, numba_fn, abstract_eval_fn_always))

    def _primitive_bind(*args):
        result = primitive.bind(*args)
        output_shapes = abstract_eval_fn(*args)
        # Special-casing when only a single tensor is returned.
        if not isinstance(output_shapes, collections.abc.Collection):
            assert len(result) == 1
            return result[0]
        else:
            return result

    if batching_fn is None:
        batching.primitive_batchers[primitive] = partial(
            _naive_batching, _primitive_bind
        )
    else:
        batching.primitive_batchers[primitive] = batching_fn

    xla.backend_specific_translations["cpu"][primitive] = partial(
        _xla_translation, numba_fn, abstract_eval_fn_always
    )
    # mlir.register_lowering(
    #     primitive,
    #     partial(
    #         _mlir_lowering,
    #         numba_fn,
    #         jit_kwargs={"nopython": nopython},
    #     ),
    #     platform="cpu",
    # )

    # jax2tf.tf_impl_with_avals[primitive] = jax2tf._convert_jax_impl(primitive)

    return _primitive_bind
