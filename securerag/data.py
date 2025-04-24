import json
import random
import torch

from securerag.profiler import profiler


def load(path: str, cfg) -> dict:
    examples = []
    with open(path, "r") as file:
        json_data = json.load(file)
        for k, example in enumerate(json_data):
            if k >= cfg.load_size:
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
    ):
        self.data = data
        self.n_context = n_context
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if "target" in example:
            target = example["target"]
            return target + " </s>"
        elif "answers" in example:
            return random.choice(example["answers"]) + " </s>"
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        target = self.get_target(example)

        if "ctxs" in example and self.n_context is not None:
            contexts = example["ctxs"][: self.n_context]
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


def encoder_batch(batch_text_passages, tokenizer, max_length):
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

    @profiler("RAGSequenceCollator")
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
                    prefix
                    + doc_title
                    + title_sep
                    + doc_text
                    + doc_sep
                    + input_string
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
        passage_ids, passage_masks = encoder_batch(
            text_passages, self.tokenizer, self.text_maxlength
        )
        questions = [[ex["question"]] for ex in batch]
        question_ids, question_masks = encoder_batch(
            questions, self.tokenizer, self.text_maxlength
        )
        scores = torch.stack([ex["scores"] for ex in batch])

        return (
            index,
            target_ids,
            target_mask,
            question_ids,
            question_masks,
            passage_ids,
            passage_masks,
            scores
        )


class FiDT5Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    @profiler("FiDT5Collator")
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
        passage_ids, passage_masks = encoder_batch(
            text_passages, self.tokenizer, self.text_maxlength
        )

        return (index, target_ids, target_mask, passage_ids, passage_masks)


class SecureRAGCollator(object):
    def __init__(
        self, text_maxlength, tokenizer, answer_maxlength=20, private_passage_ratio=0.1
    ):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.privateRatio = private_passage_ratio

    @profiler("SecureRAGCollator")
    def __call__(self, batch):
        assert batch[0]["target"] != None
        index = torch.tensor([ex["index"] for ex in batch])
        question = [ex["question"] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True if self.answer_maxlength > 0 else False,
        )
        questionIds = question["input_ids"]
        questionMask = question["attention_mask"].bool()
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

        privatePassages = []
        publicPassages = []
        for example in batch:
            if example["passages"] is None:
                return example["passages"]
            size = len(example["passages"])
            privateSize = int(size * self.privateRatio)
            privatePassages.append(
                (
                    example["question"] + " " + t
                    for t in example["passages"][0:privateSize]
                )
            )
            publicPassages.append(
                (
                    example["question"] + " " + t
                    for t in example["passages"][privateSize:size]
                )
            )
        # passage_ids size is (batchSize * passageSize * textMaxLength)
        passage_ids, passage_masks = encoder_batch(
            publicPassages, self.tokenizer, self.text_maxlength
        )
        privatePassageIds, privatePassageMasks = encoder_batch(
            privatePassages, self.tokenizer, self.text_maxlength
        )

        return (
            index,
            target_ids,
            target_mask,
            passage_ids,
            passage_masks,
            privatePassageIds,
            privatePassageMasks,
            questionIds,
        )
