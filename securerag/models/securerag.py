from calendar import c
from copy import deepcopy
import math
import torch
from torch.nn import functional
import torch.nn as nn
import transformers
from securerag.data import BatchData
from securerag.models import OutsourcingSecureModel
from securerag.models.rag_seq import RAGSequence
from securerag.models.fid import FiDT5
from securerag.profiler import profiler
from securerag.models.utils import merge_tensor
from securerag.utils import add_metric, clear_metric, get_metric

ENABLE_DEV = False


class SecureRAG(nn.Module):
    def __init__(self, fidt5: FiDT5):
        super().__init__()
        self.fidt5 = fidt5.to("cuda")
        if ENABLE_DEV:
            self.fidt5_p = deepcopy(fidt5).to("cuda")
        else:
            self.fidt5_p = deepcopy(fidt5).to("cpu")
            self.fidt5_p = OutsourcingSecureModel(self.fidt5_p)
        # TODO need to confirm the pad token id
        self.pad = fidt5.config.pad_token_id
        self.bos = fidt5.config.bos_token_id
        self.eta = 0.01

    def __del__(self):
        pfs = torch.Tensor(clear_metric("pri_fusion_size"))
        print(f"pri_fusion_size mean: {pfs.float().mean().item()}")

    def set_eta(self, val: int):
        self.eta = val

    @property
    def t5_p(self):
        return super(FiDT5, self.fidt5_p)

    def generate(
        self,
        context_ids,  # (bsz, docs, ndim)
        context_ids_private,
        attention_mask,
        attention_mask_private,
        doc_scores,  # (bsz, docs)
        doc_scores_private,
        max_length,
        output_all_ans=False,
        **kwargs,
    ):
        context_ids_all = torch.cat((context_ids, context_ids_private), dim=1)
        attention_mask_all = torch.cat((attention_mask, attention_mask_private), dim=1)
        doc_scores_all = torch.cat((doc_scores, doc_scores_private), dim=-1)
        doc_scores_all = torch.softmax(doc_scores_all, -1)
        public_scores = doc_scores_all[:, : doc_scores.size(1)]
        private_scores = doc_scores_all[:, doc_scores.size(1) :]
        if ENABLE_DEV:
            doc_scores_all = doc_scores_all.to("cuda")
            public_scores = public_scores.to("cuda")
            private_scores = private_scores.to("cuda")
            context_ids_all = context_ids_all.to("cuda")
            attention_mask_all = attention_mask_all.to("cuda")
            context_ids_private = context_ids_private.to("cuda")
            attention_mask_private = attention_mask_private.to("cuda")

        print(f"scores: {doc_scores_all}")

        def adaptive_passage_selection():
            enable_eta = True
            c_size = context_ids.size(1)
            cp_size = context_ids_private.size(1)
            total_size = c_size + cp_size
            k = c_size + cp_size
            alpha = c_size / k
            beta = cp_size / k
            m = alpha * (1 - beta) + beta * (1 - alpha) + alpha * beta
            w_pub = beta * (alpha - 1) * (1 / m)
            w_pri = alpha * (beta - 1) * (1 / m)
            w_hyb = alpha * beta * (1 / m)
            # pub_fusion_size = max(int(w_hyb * total_size), 1)
            if enable_eta:
                eta_scores_all = doc_scores_all
                eta_scores_all = eta_scores_all.mul(w_hyb)  # do copy here
                # for private doc
                eta_scores_all[:, doc_scores.size(1) :].mul_(1 + w_pri)
                eta_mask = eta_scores_all > self.eta
                eta_size = max(int(eta_mask.sum(-1).float().max().item()), 1)
                _, idx = eta_scores_all.topk(dim=-1, k=eta_size)
                pri_fusion_scores = doc_scores_all.gather(dim=1, index=idx)
                # for print
                tmp = eta_scores_all
                pri_fusion_size = eta_size
                # old method:
                # alpha_scores = private_scores.sum(-1).mean()
                # beta_scores = public_scores.sum(-1).mean()
                #     alpha_scores * tmp[:, : doc_scores.size(1)]
                # )
                # # public score process
                # tmp[:, : doc_scores.size(1)].mul_(1 - beta_scores)
                # tmp[:, : doc_scores.size(1)].clamp_min_(0.0)
                # tmp_scores, idx = torch.sort(tmp, dim=-1, descending=True)
                # pri_fusion_size = max((tmp_scores > eta).sum(-1).max(), int(1))
                # idx = idx[:, :pri_fusion_size]
                # pri_fusion_scores = doc_scores_all.gather(dim=1, index=idx)
            else:
                tmp = doc_scores_all
                pri_fusion_size = max(int((w_pri + w_hyb) * total_size), 1)
                pri_fusion_scores, idx = torch.topk(tmp, dim=-1, k=pri_fusion_size)
            print(
                f"""
                pri_fusion_size: {pri_fusion_size}
                pri_fusion_scores: {pri_fusion_scores.sum(-1)}
                eta_scores_all: {eta_scores_all}
                eta_mask: {eta_mask}
                idx: {idx}
                """
            )
            print(f"tmp: {tmp.gather(dim=1, index=idx)}")
            add_metric("pri_fusion_size", pri_fusion_size)

            idx = idx.unsqueeze(-1).expand(-1, -1, context_ids_all.size(-1))
            c_hat = context_ids_all.gather(dim=1, index=idx)
            c_hat_masks = attention_mask_all.gather(dim=1, index=idx)
            return (
                context_ids,
                attention_mask,
                public_scores,
                c_hat,
                c_hat_masks,
                pri_fusion_scores,
            )

        (
            context_ids,
            attention_mask,
            public_scores,
            context_ids_private,
            attention_mask_private,
            private_scores,
        ) = adaptive_passage_selection()

        # public fusion
        fidt5 = self.fidt5
        (y1, y1_logits) = fidt5.generate(
            input_ids=context_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            max_length=max_length,
            return_scores=True,
            **kwargs,
        )

        if ENABLE_DEV:
            y1 = y1.to("cuda")
            y1_logits = [x.to("cuda") for x in y1_logits]
        else:
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
        rank_prob_y1 = self.get_rank_prob(y1, y1_logits, public_scores.sum(dim=1))
        rank_prob_y2_p = self.get_rank_prob(
            y2_p, y2_p_logits, private_scores.sum(dim=1)
        )
        prediction = (rank_prob_y1 > rank_prob_y2_p).unsqueeze(1)
        print(f"prediction: {prediction}")
        max_len = max(y1.size(-1), y2_p.size(-1))
        pad_value = self.pad
        y1 = functional.pad(y1, (0, max_len - y1.size(-1), 0, 0), value=pad_value)
        y2_p = functional.pad(y2_p, (0, max_len - y2_p.size(-1), 0, 0), value=pad_value)
        y = torch.where(prediction, y1, y2_p)
        # return y2_p
        if output_all_ans:
            return (y1, y2_p)
        else:
            return y

    def eval_generate(self, batch: BatchData):
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
        eps = 1e-8
        doc_log_probs = torch.log(scores + eps)
        rank_prob = total_log_prob * eps + doc_log_probs  # (bsz,)
        return rank_prob
