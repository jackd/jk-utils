import keras

from ...register import register_jk_utils_serializable

ops = keras.ops


@register_jk_utils_serializable
class TorchSGD(keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        *,
        momentum: float = 0.0,
        l2_factor: float = 0.0,
        # dampening: float = 0.0,
        nesterov: bool = False,
        name="TorchSGD",
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.momentum = momentum
        # self.dampening = dampening
        self.l2_factor = l2_factor
        self.nesterov = nesterov

    def get_config(self):
        config = super().get_config()
        config.update(
            # dampening=self.dampening,
            l2_factor=self.l2_factor,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        return config

    def build(self, variables):
        if self.built:
            return
        super().build(variables)
        self.momentums = []
        if self.momentum != 0:
            for variable in variables:
                self.momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=variable, name="momentum"
                    )
                )

    def update_step(self, gradient, variable: keras.Variable, learning_rate):
        dtype = variable.dtype
        learning_rate = ops.cast(learning_rate, dtype)

        d_p = ops.cast(gradient, dtype)
        if self.l2_factor != 0:
            d_p = d_p + variable * ops.cast(self.l2_factor, dtype)

        if self.momentum != 0:
            momentum = ops.cast(self.momentum, dtype)
            var_index = self._get_variable_index(variable)
            buf: keras.Variable = self.momentums[var_index]
            # buf_value = momentum * buf + (1 - self.dampening) * d_p
            buf_value = momentum * buf + d_p
            self.assign(buf, buf_value)

            if self.nesterov:
                d_p = d_p + momentum * buf_value
            else:
                d_p = buf_value

        self.assign_sub(variable, d_p * learning_rate)
