import cProfile
import pstats
import time
import unittest

import torch.utils.data.dataloader
from torch.profiler import record_function
import transformers
import securerag
from securerag import data
import securerag.eval
import securerag.models

from securerag.models import FiDT5, RAGSequence, OutsocringSecureModel
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from test.test_base import TestConfigLoggerBase


NONDEBUG = False
ENABLE_PROFILER = True
ENABLE_C_PROFILER = False


logger = None


class TestModelBase(TestConfigLoggerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.config.device = "cpu"
        cls.config.n_context = 10
        cls.config.batch_size = 1

    def setUp(self):
        super().setUp()
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
            if ENABLE_C_PROFILER:
                # cProfile
                self.pr = cProfile.Profile()
                self.pr.enable()

    def tearDown(self):
        if ENABLE_PROFILER:
            # torch profiler
            self.profiler.stop()
            print(
                self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=15
                )
            )
            print(
                self.profiler.key_averages().table(
                    sort_by="cpu_time_total", row_limit=15
                )
            )
            if ENABLE_C_PROFILER:
                # cProfile
                self.pr.disable()
                pstats.Stats(self.pr).sort_stats("time").print_stats(15)


class TestFIDT5(TestModelBase):
    def setUp(self):
        # prepare model
        model_cls = securerag.models.FiDT5
        model_path = self.config.generator_model_path
        self.model: securerag.models.FiDT5 = model_cls.from_pretrained(model_path)
        self.model = self.model.to(self.config.device)
        self.model.eval()
        # prepare data
        path = "data/open_domain_data/NQ/dev.json"
        datas = data.load(path=path, cfg=self.config)
        dataset = data.Dataset(data=datas, n_context=self.config.n_context)
        self.tokenizer: transformers.T5Tokenizer = (
            transformers.T5Tokenizer.from_pretrained("t5-base", return_dict=False)
        )
        data_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=data.FiDT5Collator(
                tokenizer=self.tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
            ),
        )
        self.record1 = next(iter(data_loader))
        super().setUp()

    @unittest.skipIf(NONDEBUG, "including within others")
    def test_config(self):
        assert self.config.n_context == 100
        self.config.n_context = 10
        assert self.config.n_context == 10

    @unittest.skipIf(NONDEBUG, "including within others")
    def test_wrap(self):
        old = self.model.state_dict()
        print(old)
        self.model.unwrap_encoder()
        new = self.model.state_dict()
        print(new)
        self.model.save_pretrained("t5")

    def test_generate(self):
        # generate
        with torch.no_grad():
            (
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                scores,
            ) = (
                self.record1.question_ids,
                self.record1.question_masks,
                self.record1.passage_ids,
                self.record1.passage_masks,
                self.record1.scores,
            )
            start = time.time()
            output = self.model.generate(
                input_ids=context_ids.to(self.config.device),
                attention_mask=context_masks.to(self.config.device),
                max_length=self.config.answer_maxlength,
            )
            ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")

    @unittest.skipIf(NONDEBUG, "including within others")
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

    def test_outsocring_model(self):
        assert self.config.device == "cpu", "make sure that device is cpu!"
        self.model = OutsocringSecureModel(self.model)
        self.test_generate()


class TestRAGSequence(TestModelBase):
    def setUp(self):
        # prepare model
        with record_function("prepare_model"):
            model_cls: RAGSequence = RAGSequence
            checkpoint_path = "models/rag-sequence-nq"
            # retriever = transformers.RagRetriever.from_pretrained(
            #     checkpoint_path, n_docs=self.config.n_context
            # )
            self.model: RAGSequence = model_cls.from_pretrained(
                checkpoint_path, n_docs=self.config.n_context
            ).to(self.config.device)
            self.model.eval()
        # prepare data
        with record_function("prepare_data"):
            path = "data/open_domain_data/NQ/debug.json"
            datas = data.load(path=path, cfg=self.config)
            dataset = data.Dataset(data=datas, n_context=self.config.n_context)
            self.tokenizer = transformers.RagTokenizer.from_pretrained(
                checkpoint_path, return_dict=False
            ).generator
            data_loader = torch.utils.data.dataloader.DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.RAGSequenceCollator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                ),
            )
            self.record1 = next(iter(data_loader))
        return super().setUp()

    def test_generate(self):
        device = self.config.device
        # generate
        with record_function("generate"), torch.no_grad():
            (
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                scores,
            ) = (
                self.record1.question_ids,
                self.record1.question_masks,
                self.record1.passage_ids,
                self.record1.passage_masks,
                self.record1.scores,
            )

            question_ids = question_ids.to(device).squeeze(1)
            question_masks = question_masks.to(device).squeeze(1)
            context_ids = context_ids.to(device).view(-1, context_ids.size(-1))
            context_masks = context_masks.to(device).view(-1, context_masks.size(-1))
            scores = scores.to(device).view(-1, scores.size(-1))
            start = time.time()
            output = self.model.generate(
                input_ids=question_ids,
                attention_mask=question_masks,
                context_input_ids=context_ids,
                context_masks=context_masks,
                doc_scores=scores,
                max_length=50,
            )
            ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")


class TestRAGSequenceT5(TestModelBase):
    def setUp(self):
        # prepare model
        with record_function("prepare_model"):
            generator_cls = transformers.T5ForConditionalGeneration
            generator_path = "models/t5"
            generator = generator_cls.from_pretrained(generator_path)
            model_cls: RAGSequence = RAGSequence
            checkpoint_path = "models/rag-sequence-nq"
            self.model: RAGSequence = model_cls.from_pretrained(
                checkpoint_path, n_docs=self.config.n_context
            ).to(self.config.device)
            self.model.rag.generator = generator.to(self.config.device)
            self.model.eval()
        # prepare data
        with record_function("prepare_data"):
            path = "data/open_domain_data/NQ/debug.json"
            datas = data.load(path=path, cfg=self.config)
            dataset = data.Dataset(data=datas, n_context=self.config.n_context)
            self.tokenizer: transformers.T5Tokenizer = (
                transformers.T5Tokenizer.from_pretrained("t5-base", return_dict=False)
            )
            data_loader = torch.utils.data.dataloader.DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.FiDT5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                ),
            )
            self.record1 = next(iter(data_loader))
        return super().setUp()

    def test_generate(self):
        device = self.config.device
        # generate
        with record_function("generate"), torch.no_grad():
            (
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                scores,
            ) = (
                self.record1.question_ids,
                self.record1.question_masks,
                self.record1.passage_ids,
                self.record1.passage_masks,
                self.record1.scores,
            )
            question_ids = question_ids.to(device).squeeze(1)
            question_masks = question_masks.to(device).squeeze(1)
            context_ids = context_ids.to(device).view(-1, context_ids.size(-1))
            context_masks = context_masks.to(device).view(-1, context_masks.size(-1))
            scores = scores.to(device).view(-1, scores.size(-1))
            start = time.time()
            output = self.model.generate(
                input_ids=question_ids,
                attention_mask=question_masks,
                context_input_ids=context_ids,
                context_masks=context_masks,
                doc_scores=scores,
                max_length=50,
            )
            ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")

    def test_outsocring_model(self):
        assert self.config.device == "cpu", "make sure that device is cpu!"
        self.model = OutsocringSecureModel(self.model)
        self.test_generate()

