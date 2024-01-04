import keras

from ..backend import complex as _complex_backend


class Complex(keras.Operation):
    def compute_output_spec(self, real, imag):
        assert real.dtype == imag.dtype, (real.dtype, imag.dtype)
        assert real.shape == imag.shape, (real.shape, imag.shape)
        shape = real.shape
        dtype = real.dtype
        if dtype == "float32":
            return keras.KerasTensor(shape, dtype="complex64")
        if dtype == "float64":
            return keras.KerasTensor(shape, dtype="complex128")
        raise ValueError(
            f"Complex requires areguments with dtype float32 or float64, got {dtype}"
        )

    def call(self, real, imag):
        return _complex_backend.complex(real, imag)


def complex(real, imag):
    return Complex()(real, imag)
