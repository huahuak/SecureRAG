import torch
import torch.nn as nn
import transformers
from securerag.models.rag_seq import RAGSequence
from securerag.models.fid import FiDT5
from securerag.profiler import profiler
from securerag.models.utils import merge_tensor


class SecureRAG(nn.Module):
    def __init__(self, fidt5: FiDT5):
        super().__init__()
        self.fidt5 = fidt5
        # TODO the privaet fidt5 need to deepcopy from fidt5.
        self.fidt5_p = fidt5
        # TODO need to confirm the pad token id
        self.pad = fidt5.config.pad_token_id
        self.bos = fidt5.config.bos_token_id

    @property
    def t5_p(self):
        return super(FiDT5, self.fidt5_p)

    def generate(
        self,
        context_ids,
        context_ids_private,
        attention_mask,
        attention_mask_private,
        doc_scores,
        doc_scores_private,
        max_length,
        **kwargs
    ):
        # public fusion
        fidt5 = self.fidt5
        (y1, y1_logits) = fidt5.generate(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            return_scores=True,
            **kwargs,
        )
        # private fusion
        fidt5_p = self.fidt5_p
        (y2_p, y2_p_logits) = fidt5_p.generate(
            input_ids=context_ids_private,
            attention_mask=attention_mask_private,
            max_length=max_length,
            return_scores=True,
            **kwargs,
        )
        # rank_loss
        rank_loss_y1 = self.rank_loss(y1, y1_logits, doc_scores.sum(dim=1))
        rank_loss_y2_p = self.rank_loss(y2_p, y2_p_logits, doc_scores_private.sum(dim=1))
        prediction = (rank_loss_y1 > rank_loss_y2_p).unsqueeze(1)
        y = torch.where(prediction, y1, y2_p)
        return y

    @staticmethod
    def rank_loss(ans, logits, scores, remove_first_token=True):
        """rank ans by doc scores and logits, bigger is better.

        Args:
            ans: (bsz, seq_len)
            logits: (seq_len, bsz, vocab_size)
            scores: (bsz)

        Returns:
            rank_prob: (bsz,)
        """
        logits = torch.stack(logits, dim=0)  # (seq_len, bsz, vocab_size)
        if remove_first_token:
            ans = ans[:, 1:]
        logits = logits.permute(1, 0, 2)  # (bsz, seq_len, vocab_size)
        log_probs = torch.nn.functional.log_softmax(
            logits, dim=-1
        )  # (bsz, seq_len, vocab_size)

        # multiply logits
        ans_idx = ans.unsqueeze(-1)  # (bsz, seq_len, 1)
        token_log_probs = log_probs.gather(dim=2, index=ans_idx).squeeze(
            -1
        )  # (bsz, seq_len)
        total_log_prob = token_log_probs.sum(dim=1)  # (bsz,)

        # multiply doc scores (use log_softmax for log-prob space)
        doc_log_probs = torch.nn.functional.log_softmax(scores, dim=0)  # (bsz,)
        rank_prob = total_log_prob + doc_log_probs  # (bsz,)

        return rank_prob
