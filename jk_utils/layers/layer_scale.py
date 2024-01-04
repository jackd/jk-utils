import keras

from ..register import register_jk_utils_serializable


@register_jk_utils_serializable
class LayerScale(keras.layers.Layer):
    def __init__(
        self,
        axis: int = -1,
        scale_initializer=1e-6,
        scale_regularizer=None,
        scale_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        if isinstance(scale_initializer, (int, float)):
            scale_initializer = keras.initializers.Constant(scale_initializer)
        else:
            scale_initializer = keras.initializers.get(scale_initializer)
        self.scale_initializer = scale_initializer
        self.scale_regularizer = keras.regularizers.get(scale_regularizer)
        self.scale_constraint = keras.constraints.get(scale_constraint)
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "scale_initializer": keras.utils.serialize_keras_object(
                    self.scale_initializer
                ),
                "scale_regularizer": keras.utils.serialize_keras_object(
                    self.scale_regularizer
                ),
                "scale_constraint": keras.utils.serialize_keras_object(
                    self.scale_constraint
                ),
            }
        )
        return config

    def build(self, input_shape):
        if self.built:
            return
        self.scale = self.add_weight(
            shape=(input_shape[self.axis],),
            regularizer=self.scale_regularizer,
            initializer=self.scale_initializer,
            constraint=self.scale_constraint,
            name="scale",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape
