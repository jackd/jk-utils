import torch


def ema(x: torch.Tensor, factors: torch.Tensor, axis: int):
    raise NotImplementedError("TODO")


def segment_ema(
    x: torch.Tensor, factors: torch.Tensor, segment_ids: torch.Tensor, axis: int
):
    raise NotImplementedError("TODO")
