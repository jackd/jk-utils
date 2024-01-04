import unittest

import numpy as np
from absl.testing import parameterized

from jk_utils.ops import static as static_ops
from jk_utils.testing import TestCase


class StaticOpsTest(TestCase, parameterized.TestCase):
    @parameterized.product(axis=(0, 1, -1), use_weights=(False, True))
    def test_static_bincount(
        self, seed: int = 0, length=3, shape=(5, 7), axis=0, use_weights=False
    ):
        rng = np.random.default_rng(seed=seed)
        x = rng.integers(0, length, size=shape)

        weights = rng.uniform(size=shape) if use_weights else None
        actual = static_ops.static_bincount(x, weights, length, axis=axis)
        expected_shape = list(shape)
        expected_shape[axis] = length

        self.assertAllEqual(actual.shape, expected_shape)

        expected = np.zeros(expected_shape)
        if axis == 0:
            for i in range(shape[1]):
                expected[:, i] = np.bincount(
                    x[:, i], weights[:, i] if use_weights else None, minlength=length
                )
        else:
            assert axis in (-1, 1), axis
            for i in range(shape[0]):
                expected[i] = np.bincount(
                    x[i], weights[i] if use_weights else None, minlength=length
                )
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
    # StaticOpsTest().test_static_bincount()
