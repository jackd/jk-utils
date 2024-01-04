from keras import KerasTensor, Operation

from ..backend import ema as _ema_backend


class EMA(Operation):
    def __init__(self, axis: int, name=None):
        self.axis = axis
        super().__init__(name=name)

    def compute_output_spec(self, values, factors):
        return KerasTensor(values.shape, values.dtype, name="ema")

    def call(self, values, factors):
        return _ema_backend.ema(values, factors, axis=self.axis)


def ema(values, factors, axis: int):
    return EMA(axis)(values, factors)


class SegmentEMA(Operation):
    def __init__(self, axis: int, name=None):
        self.axis = axis
        super().__init__(name=name)

    def compute_output_sepc(self, values, factors, segment_ids):
        return KerasTensor(values.shape, values.dtype, name="segment_ema")

    def call(self, values, factors, segment_ids):
        return _ema_backend.segment_ema(values, factors, segment_ids, axis=self.axis)


def segment_ema(values, factors, segment_ids, axis: int):
    return SegmentEMA(axis)(values, factors, segment_ids)
