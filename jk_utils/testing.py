import unittest

import keras
import numpy as np


class TestCase(unittest.TestCase):
    def assertAllClose(self, actual, desired, rtol=1e-7, atol=0):
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(actual),
            keras.ops.convert_to_numpy(desired),
            rtol=rtol,
            atol=atol,
        )

    def assertAllEqual(self, actual, desired):
        np.testing.assert_equal(
            keras.ops.convert_to_numpy(actual), keras.ops.convert_to_numpy(desired)
        )
