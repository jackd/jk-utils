import keras

from jk_utils.layers.explicit_mask import explicit_mask
from jk_utils.layers.nan_mask import NanMask

x = keras.Input((3,))
out = NanMask()(x)
print(out._keras_mask)

m = keras.Input((3,), dtype=bool)

out = explicit_mask(x, m)
print(out._keras_mask)
