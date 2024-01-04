import typing as tp

import torch


def static_bincount(
    x: torch.Tensor,
    weights: tp.Optional[torch.Tensor] = None,
    length: tp.Optional[int] = None,
    axis: int = 0,
) -> torch.Tensor:
    if len(x.shape) != 1:
        raise NotImplementedError(
            "Only rank-1 static_bincount implemented in torch, "
            f"but x has shape {x.shape}"
        )
    if axis < 0:
        axis += len(x.shape)
    assert 0 <= axis < len(x.shape), (axis, x.shape)
    y = torch.bincount(x, weights, minlength=length)
    actual_length = y.shape[0]
    if actual_length < length:
        y = torch.nn.functional.pad(y, [0, length - actual_length])
    return y


def set_shape(x, shape):
    assert x.shape == shape, (x.shape, shape)
    return x
