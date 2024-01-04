import typing as tp

from keras import KerasTensor, ops

from ..backend import BackendTensor
from . import static

Tensor = tp.Union[BackendTensor, KerasTensor]


def splits_to_ids(
    splits: Tensor,
    dtype="int32",
    total: tp.Optional[int] = None,
) -> Tensor:
    size = splits.shape[0]
    if size is None:
        size = ops.shape(splits)[0]
    if total is None:
        total = splits[-1]
    shape = (total + 1,)

    row_ends = ops.slice(splits, (1,), (size - 1,))
    x = ops.scatter(
        ops.expand_dims(row_ends, 1), ops.ones(ops.shape(row_ends), dtype), shape
    )
    x = ops.slice(x, (0,), (total,))
    x = ops.cumsum(x)
    x = ops.cast(x, dtype)  # https://github.com/keras-team/keras/issues/18730
    # assert x.dtype == dtype, (x.dtype, dtype)
    return x


def lengths_to_ids(
    lengths: Tensor, dtype="int32", total: tp.Optional[int] = None
) -> Tensor:
    splits = lengths_to_splits(lengths)
    return splits_to_ids(splits, dtype, total)


def splits_to_lengths(splits: Tensor) -> Tensor:
    n = splits.shape[0] - 1
    return ops.slice(splits, (1,), (n,)) - ops.slice(splits, (0,), (n,))


def lengths_to_splits(lengths: Tensor, axis=0) -> Tensor:
    padding = [[0, 0] for _ in lengths.shape]
    padding[axis][0] = 1
    return ops.pad(ops.cumsum(lengths, axis=axis), padding)


def ids_to_lengths(ids: Tensor, num_segments: tp.Optional[int] = None) -> Tensor:
    return static.static_bincount(ids, length=num_segments)


def ids_to_splits(ids: Tensor, num_segments: int) -> Tensor:
    return lengths_to_splits(ids_to_lengths(ids, num_segments))


def repeat(
    a: Tensor, repeats: Tensor, axis: int, total: tp.Optional[int] = None
) -> Tensor:
    if axis < 0:
        axis += len(a.shape)
    return ops.take(a, lengths_to_ids(repeats, total=total), axis=axis)


def ragged_mask(x, splits, total_size: int):
    batch_size, row_size, *trailing = ops.unstack(ops.shape(x))
    row_lengths = splits_to_lengths(splits)
    offsets = row_size - row_lengths[:-1]

    base = ops.cast(
        ops.arange(total_size - 1) < batch_size * row_size - 1, splits.dtype
    )
    base = ops.pad(base, ((1, 0),))
    indices = base + ops.scatter(
        ops.expand_dims(splits[1:-1], 1), offsets, (total_size,)
    )
    indices = ops.cumsum(indices)

    return ops.take(ops.reshape(x, (batch_size * row_size, *trailing)), indices)


def valid_ragged_mask(splits, total_size: int):
    """
    Args:
        row_splits: [num_rows+1]
        total_size: int

    Returns:
        [total_size] bool mask
    """
    return ops.arange(total_size, dtype=splits.dtype) < splits[..., -1:]


class Partition:
    @classmethod
    def from_splits(cls, splits, total: tp.Optional[int] = None):
        return cls(splits=splits, total=total)

    @classmethod
    def from_lengths(cls, lengths, total: tp.Optional[int] = None):
        return cls(lengths=lengths, total=total)

    @classmethod
    def from_ids(cls, ids, num_segments: tp.Optional[int] = None):
        return cls(ids=ids, num_segments=num_segments)

    def __init__(
        self,
        *,
        splits: tp.Optional[Tensor],
        lengths: tp.Optional[Tensor],
        ids: tp.Optional[Tensor],
        total: tp.Optional[int] = None,
        num_segments: tp.Optional[int] = None,
    ):
        assert sum(0 if x is None else 1 for x in (splits, lengths, ids)) == 1, (
            splits,
            lengths,
            ids,
        )
        if ids is not None:
            dtype = ids.dtype
            assert len(ids.shape) == 1, ids.shape
            if total is None:
                (total,) = ids.shape
            else:
                assert ids.shape == (total,), (ids.shape, total)
        elif splits is not None:
            dtype = splits.dtype
            assert len(splits.shape) == 1, splits.shape
            if num_segments is None:
                num_segments = splits.shape[0] - 1
            else:
                assert splits.shape == (num_segments + 1,), (
                    splits.shape,
                    num_segments + 1,
                )
        elif lengths is not None:
            dtype = lengths.dtype
            assert len(lengths.shape) == 1, lengths.shape
            if num_segments is None:
                (num_segments,) = lengths.shape
            else:
                assert lengths.shape == (num_segments,), (lengths.shape, num_segments)
        self._splits = splits
        self._lengths = lengths
        self._ids = ids
        self._total = total
        self._num_segments = num_segments
        self._dtype = dtype

    def splits(self) -> Tensor:
        if self._splits is None:
            self._splits = lengths_to_splits(self.lengths())
        return self._splits

    def lengths(self) -> Tensor:
        if self._lengths is None:
            if self._splits is None:
                self._lengths = ids_to_lengths(self._ids, self.num_segments)
            else:
                self._lengths = splits_to_lengths(self._splits)
        return self._lengths

    def ids(self) -> Tensor:
        if self._ids is None:
            self._ids = splits_to_ids(self.splits(), self._dtype, total=self.total)
        return self._ids

    def total(self) -> tp.Optional[int]:
        return self._total

    def num_segments(self) -> tp.Optional[int]:
        return self._num_segments

    def dtype(self) -> str:
        return self._dtype
