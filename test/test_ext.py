import random

import torch
import torch.nn.functional as F
from test.test_base import TestConfigLoggerBase
from transformers.modeling_bart import Attention


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

    def test_attention(self):
        embed_dim = 768
        heads = 4
        class AttentionWrapper(Attention):
            def forward(self, query, key, key_padding_mask = None, layer_state = None, attn_mask = None, output_attentions=False):
                return super().forward(query, key, key_padding_mask, layer_state, attn_mask, output_attentions)[0]

        attn = AttentionWrapper(embed_dim=embed_dim, num_heads=heads)
        attn.eval()
        bsz = 8
        tgtlen = 4
        input = torch.rand(embed_dim * bsz * tgtlen).view(tgtlen, bsz, embed_dim)
        key = input
        output = attn(input, key)
        model = torch.jit.trace(attn, (input, key))
        model.save("attn.pt")

        
