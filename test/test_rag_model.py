import copy
import cProfile
import pstats
import random
import time
import unittest
from test.test_base import TestConfigLoggerBase

import numpy as np
import torch.utils.data.dataloader
import transformers
from datasets import metric
from ray import get
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader

import securerag
import securerag.eval
import securerag.models
from securerag import data
from securerag.models import (
    FiDT5,
    OutsourcingSecureModel,
    PAMLFiDT5,
    PAMLRAGSequence,
    RAGSequence,
    SecureRAG,
)
from securerag.profiler import Profiler as iprofiler
from securerag.utils import add_metric, get_metric, show_metric

NONDEBUG = True
ENABLE_PROFILER = False
ENABLE_C_PROFILER = False


logger = None


class TestModelBase(TestConfigLoggerBase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)  # fixed shuffle
        torch.manual_seed(42)
        super().setUpClass()
        cls.config.device = "cuda"
        cls.config.n_context = 10  # k
        cls.config.batch_size = 16
        cls.config.load_size = 1000
        cls.config.private_passage_ratio = 0.5

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        datas = data.load(path=path, size=cls.config.load_size)
        cls.dataset = data.Dataset(data=datas, n_context=cls.config.n_context)

    def setUp(self):
        super().setUp()

        # for auto eval
        self.k_values = np.linspace(5, 15, 3, dtype=int)
        self.eta_values = np.round(np.linspace(1.5, 2.5, 3), 1)
        self.d_values = np.round(np.linspace(0.1, 0.9, 5), 1)

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
        show_metric()

        # # plot
        # from securerag.plot import plotfig
        # plotfig(
        #     self.k_values,
        #     self.eta_values,
        #     self.d_values,
        #     ex=get_metric("ex"),
        #     f1=get_metric("f1"),
        #     # time=np.zeros_like(np.array(get_metric("time"))),  # for test, time is meaningless.
        #     time=get_metric("time"),
        #     f1_pub=get_metric("f1_pub"),
        #     f1_pri=get_metric("f1_pri"),
        # )
        # # calculate every run pri_fusion_size
        # pri_fusion_size_mean = []
        # pri_fusion_size = get_metric("pri_fusion_size")
        # batch_rounds = int(self.config.load_size / self.config.batch_size)
        # siz = [len(self.k_values), len(self.eta_values), len(self.d_values)]
        # for i in range(0, len(pri_fusion_size), batch_rounds):
        #     pri_fusion_size_mean.append(
        #         np.array(pri_fusion_size[i : i + batch_rounds]).mean()
        #     )
        # arr = np.array(pri_fusion_size_mean).reshape(siz)
        # print(
        #     f"private fusion size is {np.vectorize(lambda x: float(f'{x:.2g}'))(arr)}"
        # )

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

    def test_k_eval(self):
        import numpy as np

        for k in np.arange(5, 20, 5):
            self.config.n_context = k
            path = "data/open_domain_data/NQ/dev_with_scores.json"
            datas = data.load(path=path, size=self.config.load_size)
            self.dataset = data.Dataset(data=datas, n_context=self.config.n_context)
            self.data_loader = torch.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.FiDT5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                ),
            )
            start = time.time()
            securerag.eval.evaluate(
                model=self.model,
                dataset=self.dataset,
                dataloader=self.data_loader,
                tokenizer=self.tokenizer,
                cfg=self.config,
            )
            use_time = time.time() - start
            add_metric("time", use_time)
            print(
                f"ex: {get_metric('ex')}, f1: {get_metric('f1')}, time: {get_metric('time')}"
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


class TestLinearLayerOffloadingFiDT5(TestFIDT5):
    def setUp(self):
        super().setUp()
        self.model = OutsourcingSecureModel(model=self.model)


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

        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            "t5-base", return_dict=False
        )
        self.dataloader = DataLoader(
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

        return super().setUp()

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

    def test_pri_eta_comb_cmp_eval(self):
        for k in self.k_values:  # for range k
            self.config.n_context = k
            path = "data/open_domain_data/NQ/dev_with_scores.json"
            datas = data.load(path=path, size=self.config.load_size)
            self.dataset = data.Dataset(data=datas, n_context=self.config.n_context)
            for eta in self.eta_values:  # for range eta
                self.model.set_eta(eta)
                for ratio in self.d_values:  # for range d
                    self.config.private_passage_ratio = ratio
                    self.dataloader = DataLoader(
                        dataset=self.dataset,
                        batch_size=self.config.batch_size,
                        collate_fn=data.SecureRAG4T5Collator(
                            tokenizer=self.tokenizer,
                            text_maxlength=self.config.text_maxlength,
                            answer_maxlength=self.config.answer_maxlength,
                            private_passage_ratio=self.config.private_passage_ratio,
                        ),
                    )
                    timestart = time.time()
                    securerag.eval.test_evaluate(
                        model=self.model,
                        dataset=self.dataset,
                        dataloader=self.dataloader,
                        tokenizer=self.tokenizer,
                        cfg=self.config,
                    )
                    timeused = time.time() - timestart
                    add_metric("time", timeused)
                    print(f"k is {k}, eta is {eta}, d is {ratio}")
                    print(
                        f'ex: {get_metric("ex")}, f1: {get_metric("f1")}, time: {get_metric("time")}'
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


class TestEfficiencyForFiD(TestFIDT5):
    def setUp(self):
        self.config.n_context = 10  # k
        self.config.batch_size = 16
        self.config.load_size = 100

        super().setUp()

        self.cpu_model = copy.deepcopy(self.model).to("cpu")
        self.gpu_model = copy.deepcopy(self.model).to("cuda")
        self.llo_model = OutsourcingSecureModel(copy.deepcopy(self.model))
        self.mlo_model = PAMLFiDT5(copy.deepcopy(self.model))
        self.sa_model = SecureRAG(copy.deepcopy(self.model), enable_algorithm=False)
        self.sa_model.set_eta(2.5)
        self.sa_pf_model = SecureRAG(copy.deepcopy(self.model), enable_algorithm=True)
        self.sa_pf_model.set_eta(2.5)

    def test_efficiency(self):
        def do_eval(name, model, data_loader):
            with data.Profiler(name, print_time=True):
                securerag.eval.evaluate(
                    model=model,
                    dataset=self.dataset,
                    dataloader=data_loader,
                    tokenizer=self.tokenizer,
                    cfg=self.config,
                )
            show_metric()

        for d in self.d_values:
            self.config.private_passage_ratio = d
            dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.config.batch_size,
                collate_fn=data.SecureRAG4T5Collator(
                    tokenizer=self.tokenizer,
                    text_maxlength=self.config.text_maxlength,
                    answer_maxlength=self.config.answer_maxlength,
                    private_passage_ratio=self.config.private_passage_ratio,
                ),
            )
            self.mlo_model.set_private_ratio(d)
            do_eval("mlo_fidt5", self.mlo_model, self.data_loader)
            do_eval("sa_fidt5", self.sa_model, dataloader)
            do_eval("sa_pf_fidt5", self.sa_pf_model, dataloader)
        do_eval("llo_fidt5", self.llo_model, self.data_loader)
        do_eval("gpu_fidt5", self.gpu_model, self.data_loader)
        do_eval("cpu_fidt5", self.cpu_model, self.data_loader)


class TestAccuracyForFiD(TestFIDT5):
    def setUp(self):
        self.config.device = "cuda"
        self.config.n_context = 10  # k
        self.config.batch_size = 16
        self.config.load_size = 1000

        super().setUp()

        self.original_model = self.model
        self.mlo_model = PAMLFiDT5(copy.deepcopy(self.model))
        self.mlo_model.set_private_ratio(0.5)
        self.split_agg = SecureRAG(copy.deepcopy(self.model), enable_algorithm=False)
        self.split_agg.set_eta(2.5)
        self.split_agg_with_adpatpf = SecureRAG(
            copy.deepcopy(self.model), enable_algorithm=True
        )
        self.split_agg_with_adpatpf.set_eta(2.5)

    def test_accuracy(self):
        # securerag.eval.evaluate(
        #     model=self.original_model,
        #     dataset=self.dataset,
        #     dataloader=self.data_loader,
        #     tokenizer=self.tokenizer,
        #     cfg=self.config,
        # )
        #   show_metric()
        # securerag.eval.evaluate(
        #     model=self.mlo_model,
        #     dataset=self.dataset,
        #     dataloader=self.data_loader,
        #     tokenizer=self.tokenizer,
        #     cfg=self.config,
        # )
        # show_metric()
        # for model in [self.split_agg, self.split_agg_with_adpatpf]:
        for model in [self.split_agg_with_adpatpf]:
            for d in self.d_values:
                self.config.private_passage_ratio = d
                self.dataloader = DataLoader(
                    dataset=self.dataset,
                    batch_size=self.config.batch_size,
                    collate_fn=data.SecureRAG4T5Collator(
                        tokenizer=self.tokenizer,
                        text_maxlength=self.config.text_maxlength,
                        answer_maxlength=self.config.answer_maxlength,
                        private_passage_ratio=self.config.private_passage_ratio,
                    ),
                )
                securerag.eval.test_evaluate(
                    model=model,
                    dataset=self.dataset,
                    dataloader=self.dataloader,
                    tokenizer=self.tokenizer,
                    cfg=self.config,
                )
                show_metric()


class TestEtaForFiD(TestFIDT5):
    def setUp(self):
        self.config.n_context = 10  # k
        self.config.batch_size = 16
        self.config.load_size = 1000

        super().setUp()

        self.split_agg_with_adpatpf = SecureRAG(
            copy.deepcopy(self.model), enable_algorithm=True
        )

    def test_eta(self):
        def do_eval(name, model, data_loader):
            with data.Profiler(name, print_time=True):
                securerag.eval.test_evaluate(
                    model=model,
                    dataset=self.dataset,
                    dataloader=data_loader,
                    tokenizer=self.tokenizer,
                    cfg=self.config,
                )
            show_metric()

        for eta in self.eta_values:
            self.split_agg_with_adpatpf.set_eta(eta)
            for d in self.d_values:
                self.config.private_passage_ratio = d
                self.dataloader = DataLoader(
                    dataset=self.dataset,
                    batch_size=self.config.batch_size,
                    collate_fn=data.SecureRAG4T5Collator(
                        tokenizer=self.tokenizer,
                        text_maxlength=self.config.text_maxlength,
                        answer_maxlength=self.config.answer_maxlength,
                        private_passage_ratio=self.config.private_passage_ratio,
                    ),
                )
                do_eval(
                    f"sa_pf_fidt5_eta_{eta}",
                    self.split_agg_with_adpatpf,
                    self.dataloader,
                )
