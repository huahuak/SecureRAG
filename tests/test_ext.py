import random
import unittest

import torch


class TestExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(2025)
        torch.ops.load_library("libtorch_securerag.so")
        cls.srag = torch.ops.torch_securerag

    def test_copytensortosgx(self):
        tensor = torch.rand(1024).view(256, 4).to(torch.float32)
        tensorRef = self.srag.copyTensorToSGX(tensor)
        tensorFromSGX = self.srag.copyTensorFromSGX(tensorRef)
        self.assertTrue(torch.allclose(tensor, tensorFromSGX, atol=1e-4))
