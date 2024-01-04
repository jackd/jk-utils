import typing as tp

import tree
from keras import KerasTensor, Operation, ops


class VectorizedMap(Operation):
    def __init__(self, function: tp.Callable, name=None):
        self.function = function
        super().__init__(name=name)

    def compute_output_spec(self, args):
        flat_args = tree.flatten(args)
        batch_size = flat_args[0].shape[0]
        assert all(arg.shape[0] == batch_size for arg in flat_args[1:]), flat_args
        output = self.function(
            tree.map_structure(lambda arg: KerasTensor(arg.shape[1:], arg.dtype), args)
        )
        return tree.map_structure(
            lambda x: KerasTensor((batch_size, *x.shape), x.dtype), output
        )

    def call(self, args):
        return ops.vectorized_map(self.function, args)


def vectorized_map(function, args):
    return VectorizedMap(function)(args)
