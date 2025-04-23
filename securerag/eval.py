from pathlib import Path
import string
from venv import logger
import numpy as np
import regex
import torch
from .models import FiDT5
from transformers import RagSequenceForGeneration


def normalize(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_exact_match_score(answer, targets):
    normalize_answer = normalize(answer)
    return max([normalize_answer == normalize(it) for it in targets])


def evaluate(model, dataset, dataloader, tokenizer, cfg=None):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    total = 0
    exactmatch = []
    print_freq = 100 if cfg.eval_print_freq is not None else cfg.eval_print_freq
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # TODO: dispatcher
            if isinstance(model, FiDT5):
                (idx, _, _, context_ids, context_masks) = batch

                outputs = model.generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_masks.cuda(),
                    max_length=50,
                )
            elif isinstance(model, RagSequenceForGeneration):
                (
                    idx,
                    _,
                    _,
                    question_ids,
                    question_masks,
                    context_ids,
                    context_masks,
                    scores,
                ) = batch
                question_ids = question_ids.cuda().squeeze(1)
                question_masks = question_masks.cuda().squeeze(1)
                context_ids = context_ids.cuda().view(-1, context_ids.size(-1))
                context_masks = context_masks.cuda().view(-1, context_masks.size(-1))
                scores = scores.cuda().view(-1, scores.size(-1))
                outputs = model.generate(
                    input_ids=question_ids,
                    attention_mask=question_masks,
                    context_input_ids=context_ids,
                    context_masks=context_masks,
                    doc_scores=scores,
                    max_length=50,
                )
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if "answers" in example:
                    score = get_exact_match_score(ans, example["answers"])
                    exactmatch.append(score)
                total += 1

            if (i + 1) % print_freq == 0:
                log = f"Process: {i+1} / {len(dataloader)}"
                if len(exactmatch) == 0:
                    log += "| no answer to compute scores"
                else:
                    log += f" | average = {np.mean(exactmatch):.3f}"
                logger.warning(log)

    logger.warning(f"Process: total {total} | average = {np.mean(exactmatch):.3f}")
    return score, total
