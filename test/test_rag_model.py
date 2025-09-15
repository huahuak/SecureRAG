import cProfile
import pstats
import random
import time
import unittest

import torch.utils.data.dataloader
from torch.profiler import record_function
import transformers
import securerag
from securerag import data
import securerag.eval
import securerag.models

from securerag.models import (
    FiDT5,
    RAGSequence,
    OutsourcingSecureModel,
    PAMLFiDT5,
    PAMLRAGSequence,
)
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)
from securerag.profiler import profiler as iprofiler

from securerag.utils import get_metric
from test.test_base import TestConfigLoggerBase


NONDEBUG = True
ENABLE_PROFILER = False
ENABLE_C_PROFILER = False


logger = None


class TestModelBase(TestConfigLoggerBase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)  # fixed shuffle
        super().setUpClass()
        cls.config.device = "cpu"
        cls.config.n_context = 10
        cls.config.batch_size = 10
        cls.config.load_size = 100
        cls.config.private_passage_ratio = 0.9

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        datas = data.load(path=path, size=cls.config.load_size)
        cls.dataset = data.Dataset(data=datas, n_context=cls.config.n_context)

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
        # show metric
        print(get_metric("ex"))
        print(get_metric("f1"))
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
        self.tokenizer: transformers.T5Tokenizer = (
            transformers.T5Tokenizer.from_pretrained(
                "models/t5-base", return_dict=False
            )
        )
        self.data_loader = torch.utils.data.dataloader.DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=data.FiDT5Collator(
                tokenizer=self.tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
            ),
        )
        self.record1 = next(iter(self.data_loader))
        super().setUp()

    def model_info(self):
        print()

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
        self.model.save_pretrained("models/t5")

    def test_copy_tokenizer(self):
        self.tokenizer.save_pretrained("models/t5-base")

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

    # @unittest.skipIf(NONDEBUG, "including within others")
    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.data_loader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )

    @unittest.skipIf(NONDEBUG, "including within others")
    def test_outsourcing_model(self):
        assert self.config.device == "cpu", "make sure that device is cpu!"
        self.model = OutsourcingSecureModel(self.model)
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
            self.model.wrap_encoder_with_profile()
            self.model.eval()
        # prepare data
        with record_function("prepare_data"):
            self.tokenizer = transformers.RagTokenizer.from_pretrained(
                checkpoint_path, return_dict=False
            ).generator
            self.data_loader = torch.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.RAGSequenceCollator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                ),
            )
            self.record1 = next(iter(self.data_loader))
        return super().setUp()

    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.data_loader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )

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
            self.model.rag.generator = generator
            # self.model.wrap_encoder_with_profile()
            self.model.eval()
        # prepare data
        with record_function("prepare_data"):
            path = "data/open_domain_data/NQ/debug.json"
            datas = data.load(path=path, size=self.config.load_size)
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                "t5-base", return_dict=False
            )
            self.data_loader = torch.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.FiDT5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                ),
            )
            self.record1 = next(iter(self.data_loader))
        super().setUp()

    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.data_loader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )

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

    def test_outsourcing_model(self):
        assert self.config.device == "cpu", "make sure that device is cpu!"
        self.model = OutsourcingSecureModel(self.model)
        self.test_generate()


class TestRAGSequenceT5WithOutsource(TestRAGSequenceT5):
    def setUp(self):
        super().setUp()
        self.model = OutsourcingSecureModel(self.model)

    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.data_loader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )



class TestPAMLRAGSequenceT5(TestModelBase):
    def setUp(self):
        # preparemodel
        with record_function("prepare_model"):
            generator_cls = transformers.T5ForConditionalGeneration
            generator_path = "models/t5"
            generator = generator_cls.from_pretrained(generator_path)
            model_cls: RAGSequence = RAGSequence
            checkpoint_path = "models/rag-sequence-nq"
            self.model = model_cls.from_pretrained(
                checkpoint_path, n_docs=self.config.n_context
            ).to(self.config.device)
            self.model.rag.generator = generator.to(self.config.device)
            self.model.eval()
            self.model = OutsourcingSecureModel(self.model)
            # self.model: PAMLRAGSequence = PAMLRAGSequence(self.model)
        # prepare data
        with record_function("prepare_data"):
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                "t5-base", return_dict=False
            )
            self.data_loader = torch.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.SecureRAG4T5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                    private_passage_ratio=self.config.private_passage_ratio,
                ),
            )
            self.record1 = next(iter(self.data_loader))
        super().setUp()

    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.data_loader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )

    def test_generate(self):
        device = self.config.device
        with record_function("generate"), torch.no_grad():
            (
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                private_context_ids,
                private_context_masks,
                scores,
            ) = (
                self.record1.question_ids,
                self.record1.question_masks,
                self.record1.passage_ids,
                self.record1.passage_masks,
                self.record1.private_passage_ids,
                self.record1.private_passage_masks,
                self.record1.scores,
            )
            question_ids = question_ids.to(device).squeeze(1)
            question_masks = question_masks.to(device).squeeze(1)
            context_ids = context_ids.to(device).view(-1, context_ids.size(-1))
            context_masks = context_masks.to(device).view(-1, context_masks.size(-1))
            private_context_ids = private_context_ids.to(device).view(
                -1, context_ids.size(-1)
            )
            private_context_masks = private_context_masks.to(device).view(
                -1, context_masks.size(-1)
            )
            scores = scores.to(device).view(-1, scores.size(-1))
            start = time.time()
            output = self.model.generate(
                input_ids=question_ids,
                attention_mask=question_masks,
                context_input_ids=context_ids,
                context_masks=context_masks,
                private_context_input_ids=private_context_ids,
                private_context_masks=private_context_masks,
                doc_scores=scores,
                generator_tokenizer=self.tokenizer,
                max_length=50,
            )
            ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")


class TestPAMLFiDT5(TestFIDT5):
    def setUp(self):
        super().setUp()
        self.model = PAMLFiDT5(model=self.model)

    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.data_loader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )


class TestSecureRAG(TestModelBase):
    def setUp(self):
        # prepare model
        model_cls = securerag.models.FiDT5
        model_path = self.config.generator_model_path
        self.model = model_cls.from_pretrained(model_path)
        self.model = self.model.to(self.config.device)
        self.model.eval()
        #  switch to securerag from fidtt5
        from securerag.models.securerag import SecureRAG

        self.model: SecureRAG = SecureRAG(fidt5=self.model)
        self.model.set_eta(0.01)

        @iprofiler("prepare_data")
        def prepare_data():
            self.tokenizer = transformers.T5Tokenizer.from_pretrained(
                "t5-base", return_dict=False
            )
            self.dataloader = torch.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.SecureRAG4T5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                    private_passage_ratio=self.config.private_passage_ratio,
                ),
            )
            self.batch = next(iter(self.dataloader))

        prepare_data()

        super().setUp()

    def test_eval(self):
        securerag.eval.evaluate(
            model=self.model,
            dataset=self.dataset,
            dataloader=self.dataloader,
            tokenizer=self.tokenizer,
            cfg=self.config,
        )
    
    def test_eta_cmp_eval(self):
        import numpy as np
        for eta in np.arange(0.01, 0.11, 0.01):
            self.model.set_eta(eta)
            print(f"eta is {eta}.")
            securerag.eval.evaluate(
                model=self.model,
                dataset=self.dataset,
                dataloader=self.dataloader,
                tokenizer=self.tokenizer,
                cfg=self.config,
            )

    def test_pri_ratio_cmp_eval(self):
        import numpy as np
        for ratio in np.arange(0.1, 1, 0.2):
            print(f"ratio is {ratio}.")
            self.config.private_passage_ratio = ratio
            self.dataloader = torch.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.SecureRAG4T5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                    private_passage_ratio=self.config.private_passage_ratio,
                ),
            )
            securerag.eval.evaluate(
                model=self.model,
                dataset=self.dataset,
                dataloader=self.dataloader,
                tokenizer=self.tokenizer,
                cfg=self.config,
            )
        

    def test_generate(self):
        # generate
        device = self.config.device
        with record_function("generate"), torch.no_grad():
            batch = self.batch
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
            start = time.time()
            output = self.model.generate(
                context_ids=context_ids,
                context_ids_private=private_context_ids,
                attention_mask=context_masks,
                attention_mask_private=private_context_masks,
                doc_scores=scores,
                doc_scores_private=private_scores,
                max_length=50,
            )
            # (
            #     question_ids,  # bsz * 1 * dim
            #     question_masks,
            #     context_ids,
            #     context_masks,  # bsz * docs * dim
            #     private_context_ids,
            #     private_context_masks,  # bsz * docs_p * dim
            #     scores,  # bsz * docs
            # ) = (
            #     self.record1.question_ids,
            #     self.record1.question_masks,
            #     self.record1.passage_ids,
            #     self.record1.passage_masks,
            #     self.record1.private_passage_ids,
            #     self.record1.private_passage_masks,
            #     self.record1.scores,
            # )
            # question_ids = question_ids.to(device).squeeze(1)
            # question_masks = question_masks.to(device).squeeze(1)
            # context_ids = context_ids.to(device).view(-1, context_ids.size(-1))
            # context_masks = context_masks.to(device).view(-1, context_masks.size(-1))
            # private_context_ids = private_context_ids.to(device).view(
            #     -1, context_ids.size(-1)
            # )
            # private_context_masks = private_context_masks.to(device).view(
            #     -1, context_masks.size(-1)
            # )
            # scores = scores.to(device).view(-1, scores.size(-1))
            ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            print(ans)
            print(f"elapsed time : {time.time() - start: .3f} sec")
