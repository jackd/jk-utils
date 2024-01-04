from keras import backend, layers, ops

from ..register import register_jk_utils_serializable

layers.BatchNormalization


@register_jk_utils_serializable
class MaskedBatchNormalization(layers.BatchNormalization):
    def call(self, inputs, training=None, mask=None):
        input_dtype = backend.standardize_dtype(inputs.dtype)
        if mask is not None:
            inputs = ops.where(
                ops.expand_dims(mask, axis=-1), inputs, ops.zeros_like(inputs)
            )
            mask_factor = ops.cast(
                ops.prod(ops.convert_to_tensor(ops.shape(mask))), "float32"
            ) / ops.cast(ops.count_nonzero(mask), "float32")
        if input_dtype in ("float16", "bfloat16"):
            # BN is prone to overflowing for float16/bfloat16 inputs, so we opt
            # out BN for mixed precision.
            inputs = ops.cast(inputs, "float32")

        if training and self.trainable:
            mean, variance = ops.moments(
                inputs,
                axes=self._reduction_axes,
                synchronized=self.synchronized,
            )
            if mask is not None:
                mean = mean * mask_factor
                variance = variance * mask_factor
            moving_mean = ops.cast(self.moving_mean, inputs.dtype)
            moving_variance = ops.cast(self.moving_variance, inputs.dtype)
            self.moving_mean.assign(
                ops.cast(
                    moving_mean * self.momentum + mean * (1.0 - self.momentum),
                    inputs.dtype,
                )
            )
            self.moving_variance.assign(
                ops.cast(
                    moving_variance * self.momentum + variance * (1.0 - self.momentum),
                    inputs.dtype,
                )
            )
        else:
            moving_mean = ops.cast(self.moving_mean, inputs.dtype)
            moving_variance = ops.cast(self.moving_variance, inputs.dtype)
            mean = moving_mean
            variance = moving_variance

        if self.scale:
            gamma = ops.cast(self.gamma, inputs.dtype)
        else:
            gamma = None

        if self.center:
            beta = ops.cast(self.beta, inputs.dtype)
        else:
            beta = None

        outputs = ops.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )

        return ops.cast(outputs, input_dtype)
