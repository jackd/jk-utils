import typing as tp

from keras import KerasTensor, Operation

from ..backend import BackendTensor
from ..backend import static as _static_backend

Tensor = tp.Union[KerasTensor, BackendTensor]


class StaticBincount(Operation):
    def __init__(self, length: tp.Optional[int], axis: int = 0, name=None):
        self.length = length
        self.axis = axis
        super().__init__(name=name)

    def compute_output_spec(self, x, weights=None):
        shape = list(x.shape)
        shape[self.axis] = self.length
        return KerasTensor(shape, "int32" if weights is None else weights.dtype)

    def call(self, x, weights=None):
        return _static_backend.static_bincount(
            x, weights, length=self.length, axis=self.axis
        )


def static_bincount(
    x: Tensor,
    weights: tp.Optional[Tensor] = None,
    length: tp.Optional[int] = None,
    axis: int = 0,
) -> Tensor:
    return StaticBincount(length=length, axis=axis)(x, weights=weights)


class SetShape(Operation):
    def __init__(self, shape, name=None):
        super().__init__(name=name)
        self.shape = shape

    def compute_output_shape(self, x):
        return KerasTensor(self.shape, x.dtype)

    def call(self, x):
        return _static_backend.set_shape(x, self.shape)


def set_shape(x, shape=None):
    return SetShape(shape=x.shape if shape is None else shape)(x)
