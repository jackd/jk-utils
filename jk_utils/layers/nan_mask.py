from keras import layers, ops


class NanMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        return ops.where(ops.isnan(x), ops.zeros_like(x), x)

    def compute_mask(self, x, previous_mask=None):
        valid_mask = ops.logical_not(ops.isnan(x))
        if previous_mask is None:
            return valid_mask
        return ops.logical_and(previous_mask, valid_mask)
