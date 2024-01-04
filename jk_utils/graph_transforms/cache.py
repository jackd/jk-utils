import abc
import typing as tp

import keras
import tree
from keras.models import clone_model


class CachingLayer(keras.layers.Layer):
    """A layer that can create and update a cache for iterative inference.

    Implementations should implement `call_and_create_cache` and
    `call_with_cache`. They may optionally implement `call_without_cache`
    if creation of the cache in `call_and_create_cache` is expensive.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uses_cache = True

    def call(
        self,
        *args,
        cache=None,
        return_cache=None,
        current_index=None,
        max_length=None,
        **kwargs
    ):
        if cache is None:
            if return_cache:
                return self.call_and_create_cache(
                    *args, current_index=current_index, max_length=max_length, **kwargs
                )
            return self.call_without_cache(*args, **kwargs)
        assert return_cache is None or return_cache
        return self.call_with_cache(
            *args, cache=cache, current_index=current_index, **kwargs
        )

    @abc.abstractmethod
    def call_and_create_cache(
        self, *args, current_index=None, max_length=None, **kwargs
    ):
        """Get the output of this layer and create a cache.

        The returned cache may be used in subsequent calls to
        `call_with_cache`.
        """

    @abc.abstractmethod
    def call_with_cache(self, *args, cache, current_index=None, **kwargs):
        """Get the output of this layer using a previously created cache.

        This method should return *args, where args[:-1] is the normal
        output of the layer, and args[-1] is a single-tensor cache.
        """

    def call_without_cache(self, *args, **kwargs):
        """Get the output of this layer without a cache input or output.

        By default, this redirects to `call_and_create_cache` and throws
        out the `cache`. Implementers should override this method if
        there is a more optimal implementation that does not involve
        creating the cache at all.
        """
        *output, cache = self.call_and_create_cache(*args, **kwargs)
        del cache
        if len(output) == 1:
            return output[0]
        return output


def get_call_and_create_cache(model: keras.Model, current_index=None, max_length=None):
    """
    Get `call_and_create_cache` model from `call_without_cache`.
    """
    cache_outputs = []

    def clone_function(op):
        if isinstance(op, CachingLayer):

            def f(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs["return_cache"] = True
                kwargs["current_index"] = current_index
                kwargs["max_length"] = max_length
                *output, cache = op(*args, **kwargs)
                cache_outputs.append(cache)
                if len(output) == 1:
                    return output[0]
                return output

            return f
        return op

    inputs = tuple(tree.flatten(model.input)) + tuple(
        t for t in (current_index, max_length) if t is not None
    )
    if len(inputs) == 1:
        (inputs,) = inputs
    cloned = clone_model(model, inputs, clone_function=clone_function)
    output = cloned.output
    cache_output = keras.ops.stack(cache_outputs, axis=1)
    if isinstance(output, keras.KerasTensor):
        output = (output, cache_output)
    else:
        output = (*output, cache_output)
    return keras.Model(inputs, output)


def get_call_with_cache(model: keras.Model):
    """
    Get `call_with_cache` model from a `call_and_create_cache` model.
    """
    cache_output = model.output[-1]
    cache_input = keras.Input(batch_shape=cache_output.shape, dtype=cache_output.dtype)
    cache_inputs = keras.ops.unstack(cache_input, axis=1)
    # reverse order so we can pop in order
    cache_inputs = cache_inputs[-1::-1]

    def clone_function(op):
        if getattr(op, "uses_cache", False):

            def f(*args, **kwargs):
                assert kwargs["return_cache"], kwargs
                kwargs = dict(kwargs)
                kwargs["return_cache"] = False
                del kwargs["max_length"]
                return op(*args, cache=cache_inputs.pop(), **kwargs)

            return f
        return op

    inp = model.input
    inputs = (*(tree.flatten(inp)), cache_input)
    if current_index is not None:
        inputs = inputs + (current_index,)
    model = keras.Model(inputs, model.output)
    return clone_model(model, inputs, clone_function=clone_function)


def _is_tensor(x) -> bool:
    return hasattr(x, "__array__")


Tensor = tp.Any


class CachingFunctionalLayer(CachingLayer):
    """A caching layer made from other caching layers.

    Implementations should implement `call_without_cache`, which should
    be conceptually similar to a layer's standard `call` method without
    concerns for the absence, presence or creation of any caches used by
    constituent layers.

    Currently, the only condition on constituent caching layers is that
    they all produce caches of the same size such that they can be stacked.

    The main difference between a normal layer's `call` method and
    `call_without_cache` is that `call_without_cache` may be called with
    symbolic inputs (`keras.KerasTensor`s). This is used for graph
    transformations that create `call_and_create_cache` and `call_with_cache`
    implementations.
    """

    def _get_call_and_create_cache_model(
        self,
        args,
        kwargs,
    ) -> tp.Tuple[keras.Model, tp.List[Tensor]]:
        tensors = [arg for arg in tree.flatten((args, kwargs)) if _is_tensor(arg)]
        model_args, model_kwargs = tree.map_structure(
            lambda x: keras.Input(batch_shape=x.shape, dtype=x.dtype)
            if _is_tensor(x)
            else x,
            (args, kwargs),
        )
        inputs = [
            arg
            for arg in tree.flatten((model_args, model_kwargs))
            if keras.backend.is_keras_tensor(arg)
        ]
        output = self.call_without_cache(*model_args, **model_kwargs)
        model = keras.Model(inputs, output)
        call_and_create_cache_model = get_call_and_create_cache(model)
        return call_and_create_cache_model, tensors

    def call_and_create_cache(self, *args, **kwargs):
        """Get the output of this layer and create a cache.

        The returned cache may be used in subsequent calls to
        `call_with_cache`.
        """
        model, tensors = self._get_call_and_create_cache_model(args, kwargs)
        return model(tensors)

    def call_with_cache(self, *args, cache, current_index=None, **kwargs):
        """Get the output of this layer using a previously created cache.

        This method should return *args, where args[:-1] is the normal
        output of the layer, and args[-1] is a single-tensor cache.
        """
        call_and_create_cache_model, tensors = self._get_call_and_create_cache_model(
            args,
            kwargs,
        )
        call_with_cache_model = get_call_with_cache(
            call_and_create_cache_model, current_index
        )
        return call_with_cache_model((*tensors, cache))

    @abc.abstractmethod
    def call_without_cache(self, *args, **kwargs):
        """Get the output of this layer without a cache input or output.

        args and kwargs may contain symbolic tensors or backend tensors, but
        never both.
        """
        raise NotImplementedError("Abstract method")
