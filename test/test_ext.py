import random
import unittest

import torch
from test.test_base import TestConfigLoggerBase


class TestExt(TestConfigLoggerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        random.seed(2025)
        torch.ops.load_library("lib/libtorch_securerag.so")
        cls.srag = torch.ops.TorchSecureRAG
        cls.srag.openSGX()

    def test_copytensortosgx(self):
        tensor = torch.rand(1024).view(256, 4).to(torch.float32)
        tensorRef = self.srag.copyTensorToSGX(tensor)
        tensorFromSGX = self.srag.copyTensorFromSGX(tensorRef)
        self.assertTrue(torch.allclose(tensor, tensorFromSGX, atol=1e-4))
