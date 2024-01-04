import unittest

import keras
import numpy as np
from absl.testing import parameterized

from jk_utils import testing
from jk_utils.ops import ema as ema_ops


class EMATest(testing.TestCase, parameterized.TestCase):
    @parameterized.product(axis=(0, 1, -1))
    def test_ema(self, seed: int = 0, shape=(3, 7), axis=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=shape).astype("float32")
        f = rng.uniform(size=shape).astype("float32")
        actual = ema_ops.ema(x, f, axis=axis)

        expected = np.zeros_like(x)
        if axis == 0:
            expected[0] = x[0]
            for i in range(1, shape[0]):
                expected[i] = expected[i - 1] * f[i] + x[i]
        else:
            assert axis in (-1, 1)
            expected[:, 0] = x[:, 0]
            for i in range(1, shape[1]):
                expected[:, i] = expected[:, i - 1] * f[:, i] + x[:, i]
        self.assertAllClose(actual, expected, rtol=1e-6)

    @parameterized.product(axis=(0, 1, 2, -1, -2))
    def test_segment_ema(
        self, seed: int = 0, shape=(3, 7), segment_sizes=(2, 5, 11), axis=-1
    ):
        rng = np.random.default_rng(seed)
        shape = list(shape)
        if axis == -1:
            shape.append(sum(segment_sizes))
        elif axis < 0:
            shape.insert(axis + 1, sum(segment_sizes))
        else:
            shape.insert(axis, sum(segment_sizes))
        x = rng.normal(size=shape).astype("float32")
        f = rng.uniform(size=shape).astype("float32")
        segment_ids = np.repeat(
            np.arange(len(segment_sizes), dtype="int32"), segment_sizes
        )
        actual = ema_ops.segment_ema(x, f, segment_ids, axis=axis)
        sections = np.cumsum(segment_sizes)[:-1]
        xs = np.split(x, sections, axis=axis)
        fs = np.split(f, sections, axis=axis)
        expected = np.concatenate(
            [
                keras.ops.convert_to_numpy(ema_ops.ema(x, f, axis=axis))
                for x, f in zip(xs, fs)
            ],
            axis=axis,
        )
        self.assertAllClose(actual, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
    # EMATest().test_segment_ema()
