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
import securerag.profiler
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
from securerag.utils import (
    add_metric,
    delete_metric,
    dump_metric,
    get_metric,
    show_metric,
)

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
        cls.config.batch_size = 1
        cls.config.load_size = 100
        cls.config.private_passage_ratio = 0.5

        path = "data/open_domain_data/TQA/test_with_scores.json"
        datas = data.load(path=path, size=cls.config.load_size)
        cls.dataset = data.Dataset(data=datas, n_context=cls.config.n_context)

    def setUp(self):
        super().setUp()

        # for auto eval
        self.k_values = np.linspace(5, 15, 3, dtype=int)
        self.eta_values = np.linspace(2.5, 10, 4)
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
        dump_metric("tmp/TestFIDT5_TQA.json")

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
        dump_metric("tmp/TestRAGSequence_TQA.json")

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
            show_metric()

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
                        f"ex: {get_metric('ex')}, f1: {get_metric('f1')}, time: {get_metric('time')}"
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
        super().setUp()
        self.config.n_context = 10  # k
        self.config.batch_size = 1
        self.config.load_size = 100

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

        for model, model_name in zip(
            # [self.sa_model, self.sa_pf_model], ["sa_fidt5", "sa_pf_fidt5"]
            [self.sa_pf_model],
            ["sa_pf_fidt5"],
        ):
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
                # self.mlo_model.set_private_ratio(d)
                do_eval(model_name, model, dataloader)
            dump_metric(f"tmp/{model_name}_th0.7.json")
            # do_eval("mlo_fidt5", self.mlo_model, self.data_loader)
            # do_eval("sa_fidt5", self.sa_model, dataloader)
            # do_eval("sa_pf_fidt5", self.sa_pf_model, dataloader)
        # do_eval("llo_fidt5", self.llo_model, self.data_loader)
        # do_eval("gpu_fidt5", self.gpu_model, self.data_loader)
        do_eval("cpu_fidt5", self.cpu_model, self.data_loader)
        dump_metric(f"tmp/cpu_fidt5.json")


class TestAccuracyForFiD(TestFIDT5):
    def setUp(self):
        super().setUp()

        self.config.device = "cuda"
        self.config.n_context = 10  # k
        self.config.batch_size = 1
        self.config.load_size = 1000

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        datas = data.load(path=path, size=self.config.load_size)
        self.dataset = data.Dataset(data=datas, n_context=self.config.n_context)

        self.original_model = self.model
        self.mlo_model = PAMLFiDT5(copy.deepcopy(self.model))
        self.mlo_model.set_private_ratio(0.5)
        self.split_agg = SecureRAG(copy.deepcopy(self.model), enable_algorithm=False)
        self.split_agg.set_eta(5)
        self.split_agg_with_adpatpf = SecureRAG(
            copy.deepcopy(self.model), enable_algorithm=True
        )
        self.split_agg_with_adpatpf.set_eta(5)

    def test_accuracy(self):
        # data_loader = torch.utils.data.dataloader.DataLoader(
        #     dataset=self.dataset,
        #     batch_size=self.config.batch_size,
        #     collate_fn=data.FiDT5Collator(
        #         tokenizer=self.tokenizer,
        #         text_maxlength=self.config.text_maxlength,
        #         answer_maxlength=self.config.answer_maxlength,
        #     ),
        # )
        # securerag.eval.evaluate(
        #     model=self.original_model,
        #     dataset=self.dataset,
        #     dataloader=data_loader,
        #     tokenizer=self.tokenizer,
        #     cfg=self.config,
        # )
        # show_metric()
        # dump_metric("tmp/accuracy(original-k10)_nq_threshold0.7.json")

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
                dump_metric("tmp/accuray(eta5,k10)_nq_threshold0.9.json")


class TestEtaForFiD(TestFIDT5):
    def setUp(self):
        self.config.n_context = 10  # k
        self.config.batch_size = 16
        self.config.load_size = 1000

        super().setUp()

        path = "data/open_domain_data/NQ/dev_with_scores.json"
        datas = data.load(path=path, size=self.config.load_size)
        self.dataset = data.Dataset(data=datas, n_context=self.config.n_context)

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
            dump_metric(f"tmp/{name}.json")
            delete_metric()

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
                    f"sa_pf_fidt5_eta_{eta}_K_{10}",
                    self.split_agg_with_adpatpf,
                    self.dataloader,
                )


class TestLinearLayerOffloadingLimitation(TestConfigLoggerBase):
    def test_limitation(self):
        import time  # Wall-clock timing

        import matplotlib.pyplot as plt
        import numpy as np
        import torch  # PyTorch for tensor operations

        # ----- Configuration and Data Collection -----
        Ns = [768, 1024, 2048]
        benchmark_results = []
        WARMUP_RUNS = 5  # Number of runs to discard to ensure stability

        for N in Ns:
            # ----- Data Setup (CPU) -----
            A_cpu = torch.rand(16 * 10, 200, N, dtype=torch.float32)
            B_cpu = torch.rand(N, N, dtype=torch.float32)

            # ----- CPU Matmul Timing (No overhead concern, but still accurate) -----
            start = time.time()
            C_cpu = torch.matmul(A_cpu, B_cpu)
            cpu_time = time.time() - start

            # ----- GPU Warm-up Phase (Excluding Influence) -----
            # 1. Transfer B to GPU once (static matrix)
            B_gpu = B_cpu.to("cuda")

            # 2. Perform several dummy runs to trigger JIT compilation and context setup
            print(f"Warming up for N={N}...")
            for _ in range(WARMUP_RUNS):
                A_gpu_dummy = A_cpu.to("cuda")  # H2D transfer
                C_gpu_dummy = torch.matmul(A_gpu_dummy, B_gpu)  # Compute
                C_gpu_dummy.to("cpu")  # D2H transfer
                torch.cuda.synchronize()  # Wait for everything to finish

            # ----- GPU Matmul and Transfer Timing (Measuring steady-state) -----

            # Time H2D Transfer
            torch.cuda.synchronize()
            start = time.time()
            A_gpu = A_cpu.to("cuda")  # Transfer A
            torch.cuda.synchronize()
            h2d_time = time.time() - start

            # Time GPU Compute
            torch.cuda.synchronize()
            start = time.time()
            C_gpu = torch.matmul(A_gpu, B_gpu)
            torch.cuda.synchronize()
            gpu_compute_time = time.time() - start

            # Time D2H Transfer
            torch.cuda.synchronize()
            start = time.time()
            C_result = C_gpu.to("cpu")
            torch.cuda.synchronize()
            d2h_time = time.time() - start

            gpu_total_time = h2d_time + gpu_compute_time + d2h_time

            # Store (CPU_Time, H2D, Compute, D2H, GPU_Total)
            benchmark_results.append(
                (cpu_time, h2d_time, gpu_compute_time, d2h_time, gpu_total_time)
            )

        # --- PLOTTING (Remaining code is unchanged from the last styled version) ---
        plt.figure(figsize=(7, 5), dpi=300)

        # ----- X positions and bar width -----
        x_indices = np.arange(len(Ns))  # Indices for N=768, 1024, 2048
        width = 0.35
        group_spacing = 0.5  # Space between N groups

        # Positions for CPU bars (left) and GPU bars (right)
        x_cpu = x_indices - width / 2
        x_gpu = x_indices + width / 2

        # ----- Bar Labels and Colors (Matching First Example Style) -----
        gpu_colors = ["tab:olive", "tab:orange", "tab:green"]  # H2D, Compute, D2H
        gpu_labels = ["Host to Device", "GPU Compute", "Device to Host"]

        # We use the list to track which legend items have been drawn
        legend_handles = []
        legend_labels = []

        # ----- Plotting Loop for each N size -----
        for i, N in enumerate(Ns):
            cpu_t, h2d_t, gpu_c, d2h_t, gpu_t = benchmark_results[i]
            current_x_gpu = x_gpu[i]

            # --- 1. CPU Bar ---
            # Draw only the first CPU bar to get the legend handle
            if i == 0:
                cpu_bar = plt.bar(
                    x_cpu[i], cpu_t, width=width, color="tab:brown", label="CPU Compute"
                )
                legend_handles.append(cpu_bar)
                legend_labels.append("CPU Compute")
            else:
                plt.bar(x_cpu[i], cpu_t, width=width, color="tab:brown")

            # Add text label for CPU time
            plt.text(x_cpu[i], cpu_t + 0.001, f"{cpu_t:.3f}", ha="center", fontsize=12)

            # --- 2. GPU Stacked Bar ---
            gpu_values = [h2d_t, gpu_c, d2h_t]
            # Bottoms calculation must be cumulative
            gpu_bottoms = [0, h2d_t, h2d_t + gpu_c]

            for j, (val, bottom, color, label) in enumerate(
                zip(gpu_values, gpu_bottoms, gpu_colors, gpu_labels)
            ):
                # Draw the bar segment
                bar_segment = plt.bar(
                    current_x_gpu, val, bottom=bottom, width=width, color=color
                )

                # Draw the legend handle only on the first N (i=0)
                if i == 0:
                    legend_handles.append(bar_segment)
                    legend_labels.append(label)

                # Add percentage label to the segment
                plt.text(
                    current_x_gpu,
                    bottom + val / 2,
                    f"{val / gpu_t * 100:.1f}%",
                    ha="center",
                    fontsize=12,
                    color="white",
                )

            # Add text label for GPU total time
            plt.text(
                current_x_gpu, gpu_t + 0.001, f"{gpu_t:.3f}", ha="center", fontsize=12
            )

        # ----- X-axis labels -----
        plt.xticks(x_indices, [str(N) for N in Ns])

        # ----- Labels and title -----
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.xlabel("Matrix Dimension N", fontsize=12)

        # ----- Legend outside the plot to the right -----
        plt.legend(legend_handles, legend_labels, fontsize=12, loc="upper left")
        plt.grid(True, which="major", linestyle=":", color="gray", alpha=0.3)

        plt.tight_layout()
        plt.savefig("tmp.png")

    def test_print_memmv_and_flops(self):
        import pandas as pd

        # ----- Constants for robust timing -----
        Ns = [768, 1024, 2048]

        def calculate_metrics(N, batch_size=160, inner_dim=200, dtype_size=4):
            """
            A shape: (batch_size * inner_dim) x N
            B shape: N x N
            """

            # --- 1. FLOPS Calculation ---
            # Batched Matmul: (B*M) x K @ K x N -> (B*M) x N
            # Your A is (160*200) x N, B is N x N. M=1, K=N, N=N, B=160*200
            B = batch_size * inner_dim  # 32000
            M = 1
            K = N
            N_mat = N

            # FLOPS = Batch_Size * 2 * M * N_mat * K
            total_flops = B * 2 * N * N

            # --- 2. Memory Size Calculation (Bytes) ---
            # Memory = (Input A + Input B + Output C) * 4 Bytes (for float32)
            size_A = B * N * dtype_size
            size_B = N * N * dtype_size
            size_C = B * N * dtype_size

            total_bytes = size_A + size_B + size_C

            # --- 3. Arithmetic Intensity ---
            intensity = total_flops / total_bytes

            return total_flops, total_bytes, intensity

        print("--- Theoretical Matmul Analysis ---")
        data = []
        for N in Ns:
            flops, bytes_size, intensity = calculate_metrics(N)
            data.append(
                {
                    "N": N,
                    "FLOPS (G)": flops / 1e9,
                    "Memory (GB)": bytes_size / (1024**3),
                    "Intensity (FLOPS/Byte)": intensity,
                }
            )

            # ----- Create and Print Pandas DataFrame -----
            df = pd.DataFrame(data)

            # Optional: Format the numbers for cleaner display
            pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
            df["FLOPS (G)"] = df["FLOPS (G)"].map(
                lambda x: f"{x:,.2f}"
            )  # Keep GFLOPS to 2 decimal places

            print("\nTheoretical Matmul Analysis (FLOPS vs. Memory)")
            print("---------------------------------------------")
            print(f"\n{df.to_string(index=False)}\n")


class TestFiDT5ExecutionTime(TestFIDT5):
    def setUp(self):
        self.bsz = np.linspace(1, 64, 9, dtype=int)
        self.ks = np.linspace(4, 16, 13, dtype=int)
        super().setUp()

    def getDataLoader(self, bsz, k=10):
        path = "data/open_domain_data/NQ/dev_with_scores.json"
        datas = data.load(path=path, size=self.config.load_size)
        dataset = data.Dataset(data=datas, n_context=k)
        return torch.utils.data.dataloader.DataLoader(
            dataset=dataset,
            batch_size=bsz,
            collate_fn=data.FiDT5Collator(
                tokenizer=self.tokenizer,
                text_maxlength=self.config.text_maxlength,
                answer_maxlength=self.config.answer_maxlength,
            ),
        )

    def testFiDT5CPUExecutionTime(self):
        self.model = self.model.to("cpu")
        for i, batch in enumerate(self.data_loader):
            (
                idx,
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                scores,
            ) = (
                batch.index,
                batch.question_ids,
                batch.question_masks,
                batch.passage_ids,
                batch.passage_masks,
                batch.scores,
            )
            outputs = self.model.generate(
                input_ids=context_ids,
                attention_mask=context_masks,
                max_length=50,
            )
        show_metric()

    def testMultipleRoundCPUExecutionTime(self):
        for batch_size in self.bsz:
            for k in self.ks:
                self.config.batch_size = int(batch_size)
                self.config.n_context = int(k)
                self.data_loader = self.getDataLoader(
                    bsz=self.config.batch_size, k=self.config.n_context
                )
                self.testFiDT5CPUExecutionTime()
                self.plotHelper(f"batch_size-{batch_size}-k-{k}")
                dump_metric(f"tmp/cpu-batch_size-{batch_size}-k-{k}.json")
                delete_metric()

    def testMultipleRoundGPUExecutionTime(self):
        for batch_size in self.bsz:
            for k in self.ks:
                self.config.batch_size = int(batch_size)
                self.config.n_context = int(k)
                self.data_loader = self.getDataLoader(
                    bsz=self.config.batch_size, k=self.config.n_context
                )
                self.testFiDT5GPUExecutionTime()
                self.plotHelper(f"gpu-batch_size-{batch_size}-k-{k}")
                dump_metric(f"tmp/gpu-batch_size-{batch_size}-k-{k}.json")
                delete_metric()

    def plotHelper(self, name=""):
        import matplotlib.pyplot as plt

        encoder_timer = get_metric("ENCODER_TIMER")
        decoder_timer = get_metric("DECODER_TIMER")
        decoder_gen_timer = get_metric("DECODER_GENERATION_TIMER")

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axes[0].plot(encoder_timer)
        axes[0].set_title("ENCODER_TIMER")

        axes[1].plot(decoder_timer)
        axes[1].set_title("DECODER_TIMER")

        axes[2].plot(decoder_gen_timer)
        axes[2].set_title("DECODER_GENERATION_TIMER")

        plt.tight_layout()
        plt.savefig(f"tmp/pic-{name}.png")

    def testFiDT5GPUExecutionTime(self):
        self.model = self.model.to("cuda")
        for i, batch in enumerate(self.data_loader):
            (
                idx,
                question_ids,
                question_masks,
                context_ids,
                context_masks,
                scores,
            ) = (
                batch.index,
                batch.question_ids,
                batch.question_masks,
                batch.passage_ids,
                batch.passage_masks,
                batch.scores,
            )
            outputs = self.model.generate(
                input_ids=context_ids.to("cuda"),
                attention_mask=context_masks.to("cuda"),
                max_length=50,
            )
            outputs.to("cpu")
