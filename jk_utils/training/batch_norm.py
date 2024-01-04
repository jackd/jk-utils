import typing as tp

import keras
from keras.models import clone_model


class _MomentSums(keras.layers.Layer):
    def __init__(self, reduction_axes: tp.Sequence[int]):
        self.reduction_axes = reduction_axes
        super().__init__()

    def call(self, x):
        sum1 = keras.ops.sum(x, axis=self.reduction_axes)
        sum2 = keras.ops.sum(keras.ops.square(x), axis=self.reduction_axes)
        shape = keras.ops.shape(x)
        count = keras.ops.prod([shape[a] for a in self.reduction_axes])
        return sum1, sum2, count


def _compute_moments(inputs, sum1, sum2, count, ds, verbose):
    stats_model = keras.Model(inputs, (sum1, sum2, count))
    stats_model.compile(jit_compile=True)

    t1, t2, c = stats_model.predict(ds, verbose=verbose)
    t1 = t1.reshape(c.shape[0], -1).sum(0)
    t2 = t2.reshape(c.shape[0], -1).sum(0)
    count = c.sum()
    mean = t1 / count
    squared_mean = t2 / count
    variance = squared_mean - mean**2
    return mean, variance


def set_batch_norm_moments_from_dataset(
    model: keras.Model,
    dataset: tp.Iterable,
    batch_norm_classes: tp.Sequence[type] = (keras.layers.BatchNormalization,),
    *,
    verbose: bool = True,
):
    batch_norm_inputs_and_layers = []

    def clone_function(op):
        if isinstance(op, batch_norm_classes):

            def ret_func(x, *args, **kwargs):
                reduction_axes = list(range(len(x.shape)))
                del reduction_axes[op.axis]
                sum1, sum2, count = _MomentSums(reduction_axes)(x)
                batch_norm_inputs_and_layers.append((op, sum1, sum2, count))
                return op(x, *args, **kwargs)

            return ret_func

        return op

    inputs = model.input
    clone_model(model, inputs, clone_function=clone_function)
    num_layers = len(batch_norm_inputs_and_layers)
    for i, (layer, sum1, sum2, count) in enumerate(batch_norm_inputs_and_layers):
        if verbose:
            print(f"Computing moments for layer {i+1} / {num_layers}")

        mean, variance = _compute_moments(
            inputs, sum1, sum2, count, dataset, verbose=verbose
        )
        layer.moving_mean.assign(mean)
        layer.moving_variance.assign(variance)
