import unittest
import torch
from securerag.models.utils import merge_tensor


class TestMergeTensor(unittest.TestCase):
    def test_merge_same_shape(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([[7, 8, 9], [10, 11, 12]])
        pad = 0
        result = merge_tensor(a, b, pad)
        self.assertEqual(result.shape, (4, 3))
        self.assertTrue(torch.equal(result[0:2], a))
        self.assertTrue(torch.equal(result[2:4], b))

    def test_merge_different_row_counts(self):
        a = torch.tensor([[1, 2, 3]])
        b = torch.tensor([[4, 5, 6], [7, 8, 9]])
        pad = 0
        result = merge_tensor(a, b, pad)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(torch.equal(result[0:1], a))
        self.assertTrue(torch.equal(result[1:3], b))

    def test_merge_different_column_counts(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6, 7]])
        pad = -1
        result = merge_tensor(a, b, pad)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2, -1])))
        self.assertTrue(torch.equal(result[1], torch.tensor([3, 4, -1])))
        self.assertTrue(torch.equal(result[2], b[0]))

    def test_merge_empty_tensor(self):
        a = torch.empty((0, 3), dtype=torch.int64)
        b = torch.tensor([[1, 2, 3]])
        pad = 0
        result = merge_tensor(a, b, pad)
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(torch.equal(result, b))

    def test_merge_both_empty(self):
        a = torch.empty((0, 2), dtype=torch.int64)
        b = torch.empty((0, 2), dtype=torch.int64)
        pad = 0
        result = merge_tensor(a, b, pad)
        self.assertEqual(result.shape, (0, 2))

    def test_merge_with_padding_value(self):
        a = torch.tensor([[1]])
        b = torch.tensor([[2, 3]])
        pad = 99
        result = merge_tensor(a, b, pad)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 99])))
        self.assertTrue(torch.equal(result[1], torch.tensor([2, 3])))
