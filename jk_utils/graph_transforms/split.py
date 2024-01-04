import typing as tp

import keras
import tree
from keras import Function, Operation
from keras.models import clone_model
from keras.utils import is_keras_tensor


class SplitModel(tp.NamedTuple):
    preprocessor: Function
    model: keras.Model


class ModelOp(keras.Operation):
    def compute_output_spec(self, x):
        return keras.KerasTensor(x.shape, x.dtype)

    def call(self, x):
        return x


def model_op(x):
    """Force the output to be a part of the `model` component of any `SplitModel`."""
    return ModelOp()(x)


def split(model: keras.Model) -> SplitModel:
    """
    Split a model into preprocessor and trainable model components.

    The resulting `SplitModel` components should be consistent with input `model` when
    called in series, i.e. `out.model(out.preprocessor(x)) == model_in`.

    Args:
        model:

    Returns:
        `SplitModel`, made up of a `preprocessor` and `model`.
    """
    model_inputs = []
    model_input_ids = set()
    model_tensors = set()

    def clone_function(op: Operation):
        def fn(*args, **kwargs):
            outputs = op(*args, **kwargs)
            flat_inputs = [
                x for x in tree.flatten((args, kwargs)) if is_keras_tensor(x)
            ]
            flat_outputs = [x for x in tree.flatten(outputs) if is_keras_tensor(x)]

            has_model_inputs = any((id(x) in model_tensors for x in flat_inputs))

            is_model_layer = (
                isinstance(op, keras.Layer)
                and len(op.trainable_weights) > 0
                or isinstance(op, ModelOp)
            )
            if is_model_layer or has_model_inputs:
                for inp in flat_inputs:
                    inp_id = id(inp)
                    if inp_id not in model_tensors and inp_id not in model_input_ids:
                        model_input_ids.add(inp_id)
                        model_inputs.append(inp)
                model_tensors.update(id(x) for x in flat_outputs)

            return outputs

        return fn

    original_model_input = model.input
    original_model_output = model.output
    model = clone_model(model, original_model_input, clone_function)
    # restructure inputs
    model = keras.Model(
        tree.unflatten_as(original_model_input, model.inputs), model.output
    )
    model_inputs = tuple(model_inputs)
    preprocessor = Function(model.input, model_inputs)
    trainable_model = keras.Model(model_inputs, model.output)

    # clone model to remove preprocessor operations
    # https://github.com/keras-team/keras/issues/18647
    trainable_model = clone_model(
        trainable_model,
        tree.map_structure(
            lambda x: keras.Input(batch_shape=x.shape, dtype=x.dtype),
            model_inputs,
        ),
        lambda op: op,
    )
    # restructure outputs
    trainable_model = keras.Model(
        trainable_model.input,
        tree.unflatten_as(original_model_output, trainable_model.outputs),
    )
    return SplitModel(preprocessor, trainable_model)
