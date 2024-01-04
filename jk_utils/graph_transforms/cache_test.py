import unittest

import keras

from jk_utils import testing
from jk_utils.graph_transforms import cache as cache_lib


class CacheTest(testing.TestCase):
    def test_cache(self):
        class LagAndAdd(cache_lib.CachingLayer):
            def call_and_create_cache(self, x, *, current_index=None, max_length=None):
                del current_index, max_length
                lagged = keras.ops.pad(x[:, :-1], ((0, 0), (1, 0), (0, 0)))
                # always return the cache
                cache = x[:, -1:]
                return x + lagged, cache

            def call_with_cache(self, x, cache, current_index=None):
                del current_index
                updated_cache = x
                return x + cache, updated_cache

        inp = keras.Input((None, 3))
        x = LagAndAdd()(inp)
        x = LagAndAdd()(x)
        base_model = keras.Model(inp, x)
        x = keras.random.normal((5, 7, 3))
        base_out = base_model(x)

        call_and_create_cache_model = cache_lib.get_call_and_create_cache(base_model)
        call_with_cache_model = cache_lib.get_call_with_cache(
            call_and_create_cache_model
        )

        leading, cache = call_and_create_cache_model(x[:, :-1])
        trailing, updated_cache = call_with_cache_model((x[:, -1:], cache))

        self.assertAllClose(
            keras.ops.concatenate((leading, trailing), axis=1), base_out
        )

        class CompoundCache(cache_lib.CachingFunctionalLayer):
            def build(self, input_shape):
                if self.built:
                    return
                self.layer0 = LagAndAdd()
                self.layer1 = LagAndAdd()
                for layer in (self.layer0, self.layer1):
                    layer.build(input_shape)
                super().build(input_shape)

            def call_without_cache(self, x):
                residual = x
                x = self.layer0(x)
                x = self.layer1(x)
                return x + residual

        layer = CompoundCache()

        base_out = layer(x)
        leading, cache = layer(x[:, :-1], return_cache=True)
        trailing, cache = layer(x[:, -1:], cache=cache)
        self.assertAllClose(
            keras.ops.concatenate((leading, trailing), axis=1), base_out
        )


if __name__ == "__main__":
    unittest.main()
