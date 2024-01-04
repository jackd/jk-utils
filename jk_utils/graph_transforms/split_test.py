import unittest

import keras

from jk_utils import testing
from jk_utils.graph_transforms import split as split_lib


class SplitTest(testing.TestCase):
    def test_split(self):
        x = keras.Input((3,))
        y = x**2
        z = keras.layers.Dense(5)(y)
        model = keras.Model(x, z)
        split = split_lib.split(model)

        x = keras.random.normal((2, 3))
        (y,) = split.preprocessor(x)
        self.assertAllClose(y, x**2, rtol=1e-5)

        actual = split.model([y])
        expected = model(x)
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
