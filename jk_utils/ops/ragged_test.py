import tensorflow as tf
from absl.testing import parameterized

from jk_utils.ops import ragged as ragged_ops


class RaggedTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters({"axis": 0}, {"axis": 1}, {"axis": -1})
    def test_repeat(self, seed: int = 0, shape=(5, 7), axis: int = 0):
        rng = tf.random.Generator.from_seed(seed)
        a = rng.normal(shape)
        repeats = rng.uniform((shape[axis],), maxval=10, dtype=tf.int32)
        actual = ragged_ops.repeat(a, repeats, axis=axis)
        expected = tf.repeat(a, repeats, axis=axis)
        self.assertAllEqual(actual, expected)

    def test_splits_to_ids(self, seed: int = 0, num_rows: int = 5, max_cols: int = 7):
        rng = tf.random.Generator.from_seed(seed)
        row_lengths = rng.uniform((num_rows,), maxval=max_cols, dtype=tf.int32)
        partition = tf.experimental.RowPartition.from_row_lengths(row_lengths)
        splits = partition.row_splits()
        actual = ragged_ops.splits_to_ids(splits)
        expected = partition.value_rowids()
        self.assertAllEqual(actual, expected)

    def test_splits_to_lengths(
        self, seed: int = 0, num_rows: int = 5, max_cols: int = 7
    ):
        rng = tf.random.Generator.from_seed(seed)
        row_lengths = rng.uniform((num_rows,), maxval=max_cols, dtype=tf.int32)
        partition = tf.experimental.RowPartition.from_row_lengths(row_lengths)
        splits = partition.row_splits()
        actual = ragged_ops.splits_to_lengths(splits)
        expected = partition.row_lengths()
        self.assertAllEqual(actual, expected)

    def test_lengths_to_splits(
        self, seed: int = 0, num_rows: int = 5, max_cols: int = 7
    ):
        rng = tf.random.Generator.from_seed(seed)
        row_lengths = rng.uniform((num_rows,), maxval=max_cols, dtype=tf.int32)
        partition = tf.experimental.RowPartition.from_row_lengths(row_lengths)
        actual = ragged_ops.lengths_to_splits(partition.row_lengths())
        expected = partition.row_splits()
        self.assertAllEqual(actual, expected)

    def test_ragged_mask(self, seed: int = 0, num_rows: int = 5, max_cols: int = 7):
        rng = tf.random.Generator.from_seed(seed)
        x = rng.uniform((num_rows, max_cols))
        # x = tf.reshape(tf.range(num_rows * max_cols), (num_rows, max_cols))
        row_lengths = rng.uniform((num_rows,), maxval=max_cols, dtype=tf.int32)
        partition = tf.experimental.RowPartition.from_row_lengths(row_lengths)
        actual = ragged_ops.ragged_mask(
            x, partition.row_splits(), total_size=int(tf.reduce_sum(row_lengths))
        )
        expected = tf.RaggedTensor.from_tensor(x, row_lengths)
        self.assertAllEqual(actual, expected.flat_values)


if __name__ == "__main__":
    tf.test.main()
