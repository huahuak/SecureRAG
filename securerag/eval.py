from collections import Counter
from pathlib import Path
import string
from venv import logger
import numpy as np
import regex
import torch

from securerag.data import Profiler
from securerag.utils import add_metric
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


def get_f1_score(answer, targets):
    def f1_score(answer, target):
        prediction_tokens = normalize(answer).split()
        ground_truth_tokens = normalize(target).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max([f1_score(answer, target) for target in targets])


def get_exact_match_score(answer, targets):
    normalize_answer = normalize(answer)
    return max([normalize_answer == normalize(it) for it in targets])

@Profiler("eval.test_evaluate")
def test_evaluate(model, dataset, dataloader, tokenizer, cfg):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    total = 0
    exactmatch = []
    exactmatch_pub = []
    exactmatch_pri = []
    f1s = []
    f1s_pub = []
    f1s_pri = []
    print_freq = 10 if cfg.eval_print_freq is None else cfg.eval_print_freq

    def scores_handler(outputs, example, ex_list, f1_list):
        for k, output in enumerate(outputs):
            ans = tokenizer.decode(output, skip_special_tokens=True)
            example = dataset.data[idx[k]]
            ex, f1 = 0.0, 0.0
            if "answers" in example:
                ex = get_exact_match_score(ans, example["answers"])
                f1 = get_f1_score(ans, example["answers"])
            ex_list.append(ex)
            f1_list.append(f1)
            

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            device = cfg.device
            idx = batch.index
            (y, y_pub, y_pri) = model.eval_generate(batch, output_all_ans=True)
            scores_handler(y, dataset.data, exactmatch, f1s)
            scores_handler(y_pub, dataset.data, exactmatch_pub, f1s_pub)
            scores_handler(y_pri, dataset.data, exactmatch_pri, f1s_pri)

            # log
            total += 1
            if (i + 1) % print_freq == 0:
                log = f"Process: {i+1} / {len(dataloader)}"
                if len(exactmatch) == 0:
                    log += "| no answer to compute scores"
                else:
                    log += f" | average = {np.mean(exactmatch):.3f}"
                    log += f" | f1 average = {np.mean(f1s):.3f}"
                logger.warning(log)

    logger.warning(f"(test)Process: total {total} | ex average = {np.mean(exactmatch):.3f}")
    logger.warning(f"(test)Process: total {total} | f1 average = {np.mean(f1s):.3f}")

    # add metric
    add_metric("ex", np.mean(exactmatch))
    add_metric("ex_pub", np.mean(exactmatch_pub))
    add_metric("ex_pri", np.mean(exactmatch_pri))
    add_metric("f1", np.mean(f1s))
    add_metric("f1_pub", np.mean(f1s_pub))
    add_metric("f1_pri", np.mean(f1s_pri))
    

@Profiler("eval.evaluate")
def evaluate(model, dataset, dataloader, tokenizer, cfg):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    total = 0
    exactmatch = []
    f1s = []
    print_freq = 10 if cfg.eval_print_freq is None else cfg.eval_print_freq
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            device = cfg.device
            idx = batch.index

            if isinstance(model, RagSequenceForGeneration):
                (
                    question_ids,  # bsz * 1 * dim
                    question_masks,
                    context_ids,
                    context_masks,  # bsz * docs * dim
                    scores,  # bsz * docs
                ) = (
                    batch.question_ids,
                    batch.question_masks,
                    batch.passage_ids,
                    batch.passage_masks,
                    batch.scores,
                )
                question_ids = question_ids.to(device).squeeze(1)
                question_masks = question_masks.to(device).squeeze(1)
                context_ids = context_ids.to(device).view(-1, context_ids.size(-1))
                context_masks = context_masks.to(device).view(
                    -1, context_masks.size(-1)
                )
                scores = scores.to(device).view(-1, scores.size(-1))
                outputs = model.generate(
                    input_ids=question_ids,
                    attention_mask=question_masks,
                    context_input_ids=context_ids,
                    context_masks=context_masks,
                    doc_scores=scores,
                    max_length=50,
                )
            else:
                outputs = model.eval_generate(batch)

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if "answers" in example:
                    score = get_exact_match_score(ans, example["answers"])
                    f1 = get_f1_score(ans, example["answers"])
                    exactmatch.append(score)
                    f1s.append(f1)
                total += 1

            if (i + 1) % print_freq == 0:
                log = f"Process: {i+1} / {len(dataloader)}"
                if len(exactmatch) == 0:
                    log += "| no answer to compute scores"
                else:
                    log += f" | ex average = {np.mean(exactmatch):.3f}"
                    log += f" | f1 average = {np.mean(f1s):.3f}"
                logger.warning(log)

    logger.warning(f"Process: total {total} | average = {np.mean(exactmatch):.3f}")
    logger.warning(f"Process: total {total} | f1 average = {np.mean(f1s):.3f}")

    # add metric
    add_metric("ex", np.mean(exactmatch))
    add_metric("f1", np.mean(f1s))

    return score, total
