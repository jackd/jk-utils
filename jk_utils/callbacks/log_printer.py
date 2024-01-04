import keras
import numpy as np

from ..register import register_jk_utils_serializable


@register_jk_utils_serializable
class LogPrinter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch + 1} / {self.params['epochs']}")

        def to_string(v):
            return np.array2string(keras.ops.convert_to_numpy(v), precision=4)

        print(" - ".join([f"{k}: {to_string(v)}" for k, v in logs.items()]))

    def get_config(self):
        return super().get_config()
