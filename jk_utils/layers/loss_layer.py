from keras import layers, losses, ops


class LossLayer(layers.Layer):
    def __init__(self, loss: losses.Loss, scale_by_size: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.loss = losses.get(loss)
        self.supports_masking = True
        self.scale_by_size = scale_by_size

    def get_config(self):
        config = super().get_config()
        config.update(
            loss=losses.serialize(self.loss), scale_by_size=self.scale_by_size
        )
        return config

    def call(self, y_true, y_pred, sample_weight=None, y_pred_mask=None):
        if y_pred_mask is not None:
            if sample_weight is None:
                sample_weight = ops.cast(y_pred_mask, self.loss.dtype)
            else:
                sample_weight = sample_weight * ops.cast(y_pred_mask, self.loss.dtype)

        loss = self.loss(y_true, y_pred, sample_weight=sample_weight)
        if self.scale_by_size:
            loss = loss * ops.cast(ops.prod(ops.shape(y_true)), loss.dtype)
        self.add_loss(loss)
        return y_pred, loss
