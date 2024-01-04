import torch


def complex(real, imag):
    return torch.complex(real, imag)


def is_complex(x):
    return "complex" in str(x.dtype)


def exp(x):
    return torch.exp(x)
