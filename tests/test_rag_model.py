import cProfile
import pstats
import time
import unittest

import torch.utils.data.dataloader
import transformers
import securerag
from securerag import data
from securerag.config import Config
import securerag.eval
import securerag.models

from securerag.models import FiDT5, RAGSequence
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)


NONDEBUG = False
ENABLE_PROFILER = True


class TestModelBase(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        if ENABLE_PROFILER:
            # torch profile
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=tensorboard_trace_handler("./log"),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            )
            self.profiler.start()
            # cProfile
            self.pr = cProfile.Profile()
            self.pr.enable()

    def tearDown(self):
        if ENABLE_PROFILER:
            # torch profiler
            self.profiler.stop()
            print(
                self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=15)
            )
            print(
                self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=15)
            )
            # cProfile
            self.pr.disable()
            pstats.Stats(self.pr).sort_stats('time').print_stats(15)


class TestFIDT5(TestModelBase):
    def test_config(self):
        assert self.config.n_context == 100
        self.config.n_context = 10
        assert self.config.n_context == 10

    @unittest.skipIf(NONDEBUG, "including within others")
    def test_data_loader(self):
        path = "data/open_domain_data/NQ/debug.json"
        datas = data.load(path, self.config)
        dataset = data.Dataset(data=datas, n_context=self.config.n_context)
        tokenizer = transformers.T5Tokenizer.from_pretrained(
            "t5-base", return_dict=False
        )
        data_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=data.SecureRAGCollator(
                tokenizer=tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
                private_passage_ratio=self.config.private_passage_ratio,
            ),
        )
        record1 = next(iter(data_loader))
        print(record1)

    @unittest.skipIf(NONDEBUG, "including within others")
    def test_generate(self):
        # prepare model
        model_cls = securerag.models.FiDT5
        model_path = self.config.generator_model_path
        model = model_cls.from_pretrained(model_path)
        model = model.to(self.config.device)
        # prepare data
        path = "data/open_domain_data/NQ/dev.json"
        datas = data.load(path=path, cfg=self.config)
        dataset = data.Dataset(data=datas, n_context=self.config.n_context)
        tokenizer: transformers.T5Tokenizer = transformers.T5Tokenizer.from_pretrained(
            "t5-base", return_dict=False
        )
        data_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=data.FiDT5Collator(
                tokenizer=tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
            ),
        )
        record1 = next(iter(data_loader))
        # generate
        with torch.no_grad():
            (idx, _, _, context_ids, context_mask) = record1
            start = time.time()
            output = model.generate(
                input_ids=context_ids.to(self.config.device),
                attention_mask=context_mask.to(self.config.device),
                max_length=self.config.answer_maxlength,
            )
            ans = tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")

    def test_eval(self):
        # prepare model
        model_cls = securerag.models.FiDT5
        model_path = self.config.generator_model_path
        model = model_cls.from_pretrained(model_path)
        model = model.to(self.config.device)
        # prepare dataset
        path = "data/open_domain_data/NQ/dev.json"
        datas = data.load(path)
        dataset = data.Dataset(data=datas, n_context=self.config.n_context)
        tokenizer: transformers.T5Tokenizer = transformers.T5Tokenizer.from_pretrained(
            "t5-base", return_dict=False
        )
        data_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=data.FiDT5Collator(
                tokenizer=tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
            ),
        )
        # eval
        securerag.eval.evaluate(
            model=model,
            dataset=dataset,
            dataloader=data_loader,
            tokenizer=tokenizer,
            cfg=self.config,
        )


class TestRAGSequence(TestModelBase):
    def setUp(self):
        super().setUp()
        self.config.n_context = 10

    def test_generate(self):
        # prepare model
        model_cls: RAGSequence = RAGSequence
        checkpoint_path = "models/rag-sequence-nq"
        # retriever = transformers.RagRetriever.from_pretrained(
        #     checkpoint_path, n_docs=self.config.n_context
        # )
        model: RAGSequence = model_cls.from_pretrained(
            checkpoint_path, n_docs=self.config.n_context
        ).to(self.config.device)
        # prepare data
        path = "data/open_domain_data/NQ/debug.json"
        datas = data.load(path=path, cfg=self.config)
        dataset = data.Dataset(data=datas, n_context=self.config.n_context)
        tokenizer = transformers.RagTokenizer.from_pretrained(
            checkpoint_path, return_dict=False
        ).generator
        data_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=data.RAGSequenceCollator(
                tokenizer=tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
            ),
        )
        record1 = next(iter(data_loader))
        # generate
        with torch.no_grad():
            (
                idx,
                _,
                _,
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                scores,
            ) = record1
            question_ids = question_ids.cuda().squeeze(1)
            question_masks = question_masks.cuda().squeeze(1)
            context_ids = context_ids.cuda().view(-1, context_ids.size(-1))
            context_masks = context_masks.cuda().view(-1, context_masks.size(-1))
            scores = scores.cuda().view(-1, scores.size(-1))
            start = time.time()
            output = model.generate(
                input_ids=question_ids,
                attention_mask=question_masks,
                context_input_ids=context_ids,
                context_masks=context_masks,
                doc_scores=scores,
                max_length=50,
            )
            ans = tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")
