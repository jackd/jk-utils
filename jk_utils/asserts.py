import typing as tp

Tensor = tp.Any


def assert_has_rank(x: Tensor, rank: int, name: str = "x"):
    if len(x.shape) != rank:
        raise ValueError(
            f"{name} must have rank {rank} but has shape {x.shape} "
            f"(rank {len(x.shape)})"
        )


def assert_shape_compatible(
    x: Tensor, shape: tp.Tuple[tp.Union[int, None]], name: str = "x"
):
    assert_has_rank(x, len(shape), name)
    for i, (actual, expected) in enumerate(zip(x.shape, shape)):
        if expected is None:
            continue
        if actual != expected:
            raise ValueError(
                f"Expected {name}.shape[{i}] for be {expected}, got {actual}. "
                f"{name}.shape = {x.shape}, expected_shape = {shape}"
            )
