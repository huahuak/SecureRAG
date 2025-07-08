from copy import deepcopy
import torch
from torch.nn import functional
import torch.nn as nn
import transformers
from securerag.data import BatchData
from securerag.models.rag_seq import RAGSequence
from securerag.models.fid import FiDT5
from securerag.profiler import profiler
from securerag.models.utils import merge_tensor


class SecureRAG(nn.Module):
    def __init__(self, fidt5: FiDT5):
        super().__init__()
        self.fidt5 = fidt5.to("cuda")
        # TODO the privaet fidt5 need to deepcopy from fidt5.
        self.fidt5_p = deepcopy(fidt5).to("cpu")
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
        output_all_ans=False,
        **kwargs
    ):
        # public fusion
        fidt5 = self.fidt5
        (y1, y1_logits) = fidt5.generate(
            input_ids=context_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            max_length=max_length,
            return_scores=True,
            **kwargs,
        )
        y1 = y1.to("cpu")
        y1_logits = [x.to("cpu") for x in y1_logits]
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
        rank_prob_y1 = self.get_rank_prob(y1, y1_logits, doc_scores.sum(dim=1))
        rank_prob_y2_p = self.get_rank_prob(
            y2_p, y2_p_logits, doc_scores_private.sum(dim=1)
        )
        prediction = (rank_prob_y1 > rank_prob_y2_p).unsqueeze(1)
        max_len = max(y1.size(-1), y2_p.size(-1))
        pad_value = self.pad
        y1 = functional.pad(y1, (0, max_len - y1.size(-1), 0, 0), value=pad_value)
        y2_p = functional.pad(y2_p, (0, max_len - y2_p.size(-1), 0, 0), value=pad_value)
        y = torch.where(prediction, y1, y2_p)
        print(prediction)
        if output_all_ans:
            return (y1, y2_p)
        else:
            return y

    def eval_generate(self, batch: BatchData):
        device = self.cfg.device
        (
            question_ids,  # bsz * 1 * dim
            question_masks,
            context_ids,
            context_masks,  # bsz * docs * dim
            private_context_ids,
            private_context_masks,  # bsz * docs_p * dim
            scores,  # bsz * docs
            private_scores,
        ) = (
            batch.question_ids,
            batch.question_masks,
            batch.passage_ids,
            batch.passage_masks,
            batch.private_passage_ids,
            batch.private_passage_masks,
            batch.scores,
            batch.private_scores,
        )
        output = self.generate(
            context_ids=context_ids,
            context_ids_private=private_context_ids,
            attention_mask=context_masks,
            attention_mask_private=private_context_masks,
            doc_scores=scores,
            doc_scores_private=private_scores,
            max_length=50,
        )
        return output

    @staticmethod
    def get_rank_prob(ans, logits, scores, remove_first_token=True):
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
