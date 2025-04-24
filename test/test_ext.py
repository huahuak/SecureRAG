import random

import torch
import torch.nn.functional as F
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

    def test_linear(self):
        N = 1024
        indim = 1024
        outdim = 512
        linear = torch.nn.Linear(indim, outdim, True, dtype=torch.float32)
        a = torch.rand((N, indim), dtype=torch.float32)
        b = linear.weight
        c = linear.bias
        out = self.srag.secureLinear(a, b, c)
        ans = F.linear(a, b, c)
        self.assertTrue(torch.allclose(out, ans, atol=1e-4))
