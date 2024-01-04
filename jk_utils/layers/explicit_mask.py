from keras import layers, ops


class ExplicitMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, args):
        x, mask = args
        return x

    def compute_mask(self, args, previous_mask):
        _, mask = args
        if previous_mask is None:
            return mask
        return ops.logical_and(mask, previous_mask)


def explicit_mask(x, mask, **layer_kwargs):
    return ExplicitMask(**layer_kwargs)((x, mask))
