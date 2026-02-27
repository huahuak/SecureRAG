import json
import random
from audioop import reverse
from dataclasses import dataclass
from typing import Optional

import torch

from securerag.profiler import Profiler


class PreProcess:
    def __init__(self, encoder):
        self.encoder = encoder.eval()


def load(path: str, size) -> dict:
    examples = []
    with open(path, "r") as file:
        json_data = json.load(file)
        for k, example in enumerate(json_data):
            if k >= size:
                break
            if not "id" in example:
                example["id"] = k
            for c in example["ctxs"]:
                if not "score" in c:
                    c["score"] = 1.0 / (k + 1)
            examples.append(example)
    return examples


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        n_context=None,
        enable_shuffle=True,
    ):
        self.data = data
        self.n_context = n_context
        self.enable_shuffle = enable_shuffle
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if "target" in example:
            target = example["target"]
            return target + " </s>"
        elif "answers" in example:
            # return random.choice(example["answers"]) + " </s>"
            return example["answers"][0] + " </s>"
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        target = self.get_target(example)

        if "ctxs" in example and self.n_context is not None:
            contexts = example["ctxs"][: self.n_context]
            if self.enable_shuffle:
                # contexts.sort(key=lambda x: float(x["score"]), reverse=True)
                random.shuffle(contexts)
            scores = [float(c["score"]) for c in contexts]
            scores = torch.tensor(scores)
        else:
            passages, scores = None, None

        return {
            "index": index,
            "question": example["question"],
            "target": target,
            "passages": contexts,
            "scores": scores,
        }

    def sort_data(self):
        if self.n_context is None or not "score" in self.data[0]["ctxs"][0]:
            return
        for ex in self.data:
            ex["ctxs"].sort(key=lambda x: float(x["score"]), reverse=True)

    def get_example(self, index):
        return self.data[index]


@dataclass
class BatchData:
    index: Optional = None
    target_ids: Optional = None
    target_mask: Optional = None
    question_ids: Optional = None
    question_masks: Optional = None
    passage_ids: Optional = None
    passage_masks: Optional = None
    private_passage_ids: Optional = None
    private_passage_masks: Optional = None
    scores: Optional = None
    private_scores: Optional = None


def tokenizer_encode_batch2(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True,
        )
        passage_ids.append(p["input_ids"])
        passage_masks.append(p["attention_mask"])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


def tokenizer_encode_batch(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True,
        )
        passage_ids.append(p["input_ids"][None])
        passage_masks.append(p["attention_mask"][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class RAGSequenceCollator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    @Profiler("RAGSequenceCollator")
    def __call__(self, batch):
        assert batch[0]["target"] != None
        index = torch.tensor([ex["index"] for ex in batch])
        target = [ex["target"] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            title_sep = "/"
            doc_sep = "//"
            if example["passages"] is None:
                return [example["question"]]

            def cat_input_and_doc(doc_title, doc_text, input_string, prefix=None):
                if doc_title.startswith('"'):
                    doc_title = doc_title[1:]
                if doc_title.endswith('"'):
                    doc_title = doc_title[:-1]
                if prefix is None:
                    prefix = ""
                out = (
                    prefix + doc_title + title_sep + doc_text + doc_sep + input_string
                ).replace("  ", " ")
                return out

            return [
                cat_input_and_doc(
                    doc_title=p["title"],
                    doc_text=p["text"],
                    input_string=example["question"],
                )
                for p in example["passages"]
            ]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = tokenizer_encode_batch(
            text_passages, self.tokenizer, self.text_maxlength
        )
        questions = [[ex["question"]] for ex in batch]
        question_ids, question_masks = tokenizer_encode_batch(
            questions, self.tokenizer, self.text_maxlength
        )
        scores = torch.stack([ex["scores"] for ex in batch])

        return BatchData(
            index=index,
            target_ids=target_ids,
            target_mask=target_mask,
            question_ids=question_ids,
            question_masks=question_masks,
            passage_ids=passage_ids,
            passage_masks=passage_masks,
            scores=scores,
        )


class FiDT5Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    @Profiler("FiDT5Collator")
    def __call__(self, batch):
        assert batch[0]["target"] != None
        index = torch.tensor([ex["index"] for ex in batch])
        target = [ex["target"] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            question_prefix = "question:"
            title_prefix = "title:"
            passage_prefix = "context:"
            f = (
                question_prefix
                + " {} "
                + title_prefix
                + " {} "
                + passage_prefix
                + " {}"
            )
            if example["passages"] is None:
                return [example["question"]]
            return [
                f.format(example["question"], t["title"], t["text"])
                for t in example["passages"]
            ]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = tokenizer_encode_batch(
            text_passages, self.tokenizer, self.text_maxlength
        )
        questions = [[ex["question"]] for ex in batch]
        question_ids, question_masks = tokenizer_encode_batch(
            questions, self.tokenizer, self.text_maxlength
        )
        scores = torch.stack([ex["scores"] for ex in batch])

        return BatchData(
            index=index,
            target_ids=target_ids,
            target_mask=target_mask,
            question_ids=question_ids,
            question_masks=question_masks,
            passage_ids=passage_ids,
            passage_masks=passage_masks,
            scores=scores,
        )


class SecureRAG4T5Collator(object):
    def __init__(
        self, text_maxlength, tokenizer, answer_maxlength=20, private_passage_ratio=0.1
    ):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.private_ratio = private_passage_ratio

    @Profiler("SecureRAGCollator")
    def __call__(self, batch, private_ratio=None):
        if private_ratio is None:
            private_ratio = self.private_ratio
        assert batch[0]["target"] != None
        index = torch.tensor([ex["index"] for ex in batch])
        target = [ex["target"] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        public_passages = []
        private_passages = []
        for example in batch:
            public_tmp = []
            private_tmp = []
            if example["passages"] is None:
                return example["question"]
            size = len(example["passages"])
            private_size = max(1, int(size * private_ratio))
            question_prefix = "question:"
            title_prefix = "title:"
            passage_prefix = "context:"
            f = (
                question_prefix
                + " {} "
                + title_prefix
                + " {} "
                + passage_prefix
                + " {}"
            )
            for t in example["passages"][0:private_size]:
                private_tmp.append(f.format(example["question"], t["title"], t["text"]))
            for t in example["passages"][private_size:size]:
                public_tmp.append(f.format(example["question"], t["title"], t["text"]))
            public_passages.append(public_tmp)
            private_passages.append(private_tmp)

        # passage_ids_size is (batchsize * publicpassagesize * textmaxlength)
        passage_ids, passage_masks = tokenizer_encode_batch(
            public_passages, self.tokenizer, self.text_maxlength
        )
        private_passage_ids, private_passage_masks = tokenizer_encode_batch(
            private_passages, self.tokenizer, self.text_maxlength
        )
        questions = [[ex["question"]] for ex in batch]
        question_ids, question_masks = tokenizer_encode_batch(
            questions, self.tokenizer, self.text_maxlength
        )
        scores = torch.stack([ex["scores"] for ex in batch])
        bsz = len(batch)
        private_scores = scores[:bsz, :private_size]
        public_scores = scores[:bsz, private_size:]

        return BatchData(
            index=index,
            target_ids=target_ids,
            target_mask=target_mask,
            question_ids=question_ids,
            question_masks=question_masks,
            passage_ids=passage_ids,
            passage_masks=passage_masks,
            private_passage_ids=private_passage_ids,
            private_passage_masks=private_passage_masks,
            scores=public_scores,
            private_scores=private_scores,
        )
