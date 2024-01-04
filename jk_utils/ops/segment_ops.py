import typing as tp

from keras import KerasTensor, Operation

from ..backend import segment_ops as _segment_ops_backend


class _SegmentOp(Operation):
    def __init__(
        self,
        num_segments: tp.Optional[int] = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        bucket_size: tp.Optional[int] = None,
        mode=None,
        name=None,
    ):
        super().__init__(name=name)
        self.num_segments = num_segments
        self.indices_are_sorted = indices_are_sorted
        self.unique_indices = unique_indices
        self.bucket_size = bucket_size
        self.mode = mode

    def compute_output_spec(self, data, segment_ids):
        return KerasTensor((self.num_segments, *data.shape[1:]), data.dtype)


class SegmentSum(_SegmentOp):
    def call(self, data, segment_ids):
        return _segment_ops_backend.segment_sum(
            data,
            segment_ids,
            num_segments=self.num_segments,
            indices_are_sorted=self.indices_are_sorted,
            unique_indices=self.unique_indices,
            bucket_size=self.bucket_size,
            mode=self.mode,
        )


def segment_sum(
    data,
    segment_ids,
    num_segments: tp.Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: tp.Optional[int] = None,
    mode=None,
):
    return SegmentSum(
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        bucket_size=bucket_size,
        mode=mode,
    )(data, segment_ids)


class SegmentProd(_SegmentOp):
    def call(self, data, segment_ids):
        return _segment_ops_backend.segment_prod(
            data,
            segment_ids,
            num_segments=self.num_segments,
            indices_are_sorted=self.indices_are_sorted,
            unique_indices=self.unique_indices,
            bucket_size=self.bucket_size,
            mode=self.mode,
        )


def segment_prod(
    data,
    segment_ids,
    num_segments: tp.Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: tp.Optional[int] = None,
    mode=None,
):
    return SegmentProd(
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        bucket_size=bucket_size,
        mode=mode,
    )(data, segment_ids)


class SegmentMin(_SegmentOp):
    def call(self, data, segment_ids):
        return _segment_ops_backend.segment_min(
            data,
            segment_ids,
            num_segments=self.num_segments,
            indices_are_sorted=self.indices_are_sorted,
            unique_indices=self.unique_indices,
            bucket_size=self.bucket_size,
            mode=self.mode,
        )


def segment_min(
    data,
    segment_ids,
    num_segments: tp.Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: tp.Optional[int] = None,
    mode=None,
):
    return SegmentMin(
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        bucket_size=bucket_size,
        mode=mode,
    )(data, segment_ids)


class SegmentMax(_SegmentOp):
    def call(self, data, segment_ids):
        return _segment_ops_backend.segment_max(
            data,
            segment_ids,
            num_segments=self.num_segments,
            indices_are_sorted=self.indices_are_sorted,
            unique_indices=self.unique_indices,
            bucket_size=self.bucket_size,
            mode=self.mode,
        )


def segment_max(
    data,
    segment_ids,
    num_segments: tp.Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: tp.Optional[int] = None,
    mode=None,
):
    return SegmentMax(
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        bucket_size=bucket_size,
        mode=mode,
    )(data, segment_ids)
