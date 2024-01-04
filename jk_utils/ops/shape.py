from keras import Operation

from ..backend import shape as _shape_backend


class DynamicShape(Operation):
    def __init__(self, out_type: int = "int32", name=None):
        self.out_type = out_type
        super().__init__(name=name)

    def compute_output_spec(self, input):
        return KerasTensor((len(input),), dtype=self.out_type)

    def call(self, input):
        return _shape_backend.dynamic_shape(input, self.out_type)


def dynamic_shape(input, out_type="int32"):
    return DynamicShape(oout_type)(input)
