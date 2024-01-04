from keras import losses, ops, utils


def squeeze_to_same_rank(x1, x2):
    """Squeeze last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            x1 = ops.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            x2 = ops.squeeze(x2, axis=-1)
    return x1, x2


class LossFunctionWrapper(losses.Loss):
    def __init__(self, fn, reduction="sum_over_batch_size", name=None, **kwargs):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {"fn": utils.serialize_keras_object(self.fn)}
        config.update(utils.serialize_keras_object(self._fn_kwargs))
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = utils.deserialize_keras_object(config)
        return cls(**config)
