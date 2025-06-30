from textwrap import indent
import unittest
import torch
from securerag.models.securerag import SecureRAG


class TestSecureRAG (unittest.TestCase):
    def test_rank_loss_basic(self):
        bsz, seq_len, vocab_size = 2, 3, 5
        ans = torch.tensor([[0, 1, 2, 0], [0, 2, 1, 2]])
        logits = [torch.randn(bsz, vocab_size) for _ in range(seq_len)]
        scores = torch.randn(bsz)
        rank_loss = SecureRAG.rank_prob(ans, logits, scores)
        assert rank_loss.shape == (bsz,)
        assert isinstance(rank_loss, torch.Tensor)

    def test_rank_loss_values(self):
        # 构造确定性输入
        ans = torch.tensor([[0, 1], [1, 0]])
        logits = [
            torch.tensor([[2.0, 1.0], [1.0, 2.0]]),  # (bsz, vocab_size)
            torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        ]
        scores = torch.tensor([0.5, 1.5])
        # 调用rank_loss
        rank_loss = SecureRAG.rank_prob(ans, logits, scores, remove_first_token=False)
        print(rank_loss)
        assert rank_loss.shape == (2,)
        # 检查输出不是nan
        assert not torch.isnan(rank_loss).any()
    
    def test_topk(self):
        tensor1 = torch.randn(3, 8)
        tensor2 = torch.randn(3, 8)
        tensor = torch.stack([tensor1, tensor2], dim=1)
        print(tensor.shape)
        loss1 = torch.tensor([0.1, 0.2, 0.3])
        loss2 = torch.tensor([0.3, 0.2, 0.4])
        loss = torch.stack([loss1, loss2], dim=1)
        indices = torch.topk(loss, k=1, dim=1).indices
        print(indices.shape)
        assert indices.shape == (3, 1)
        tensor  = tensor[indices]
        print(tensor.shape)
        tensor = tensor.squeeze(1)
        print(tensor.shape)