import json
import time
from collections import defaultdict
from concurrent import futures
from copy import deepcopy
from typing import List, overload

import grpc
import numpy as np
import torch
import transformers
from google.protobuf import empty_pb2
from pyexpat.errors import messages
from torch.nn.functional import pad
from transformers.file_utils import ModelOutput

from securerag.config import Config
from securerag.data import (
    BatchData,
    SecureRAG4T5Collator,
    tokenizer_encode_batch,
    tokenizer_encode_batch2,
)
from securerag.models import SecureRAG
from securerag.models.fid import FiDT5
from securerag.profiler import Profiler
from securerag.rpc import messages_pb2
from securerag.rpc.messages_pb2_grpc import (
    DecoderService,
    DecoderServiceStub,
    EncoderService,
    EncoderServiceStub,
    GenerateService,
    MetricService,
    add_DecoderServiceServicer_to_server,
    add_EncoderServiceServicer_to_server,
    add_GenerateServiceServicer_to_server,
    add_MetricServiceServicer_to_server,
)
from securerag.scheduler.dependency import FusionAggregate
from securerag.scheduler.requests import Request
from securerag.utils import add_metric, clear_metric, delete_metric, show_metric


class Task:
    def __init__(self):
        self.request = None
        self.env_type = "TEE"  # default env
        self.input = None
        self.output = None
        self.dep = None
        self.is_finished = False

    def check_dep(self):
        return self.dep is None or self.dep.resolve()

    def create_native_task_from_request(req: Request):
        data = req.input_data
        input_data = {
            "index": data["index"],
            "target": data["target"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"],
            "scores": data["scores"],
        }
        task = Task()
        task.request = req
        task.env_type = "TEE"
        task.input = input_data
        return task

    @staticmethod
    def create_task_from_request(req: Request):
        data = req.input_data
        private_passage_size = req.private_passage_size
        print(f"private_passage_size: {private_passage_size}")

        public_data = {
            "index": data["index"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"][:-private_passage_size],
            "scores": data["scores"][:-private_passage_size],
        }

        private_data = {
            "index": data["index"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"][-private_passage_size:],
            "scores": data["scores"][-private_passage_size:],
        }

        public_task = None
        if len(public_data["passages"]) != 0:
            public_task = Task()
            public_task.request = req
            public_task.env_type = "GPU"
            public_task.input = public_data

        private_task = None
        if len(private_data["passages"]) != 0:
            private_task = Task()
            private_task.request = req
            private_task.env_type = "TEE"
            private_task.input = private_data

        if public_task and private_task:
            private_task.dep = FusionAggregate(public_task, private_task)
            dep_idx = private_task.dep.get_public_dep_idx()
            add_metric("required_dep_size", len(dep_idx))
            if len(dep_idx) > 0:
                public_task.input["dep_idx"] = dep_idx
            else:
                private_task.dep = None

        return (public_task, private_task)

    def create_decoder_task(pre_task, new_task=True):
        if new_task:
            task = Task()
        else:
            task = pre_task
        task.request = pre_task.request
        task.env_type = pre_task.env_type
        # extract input
        question = pre_task.input["question"]
        scores = pre_task.input["scores"]
        # extract output
        contexts = pre_task.output["contexts"]
        contexts = contexts.contiguous().view(
            contexts.size(0) * contexts.size(1), contexts.size(2)
        )
        masks = pre_task.output["context_masks"]
        masks = masks.contiguous().view(masks.size(0) * masks.size(1))
        task.input = {
            "question": question,
            "scores": scores,
            "contexts": contexts,
            "context_masks": masks,
        }
        return task

    def check_non_none(v):
        if v is None:
            return False
        if isinstance(v, str) and len(v) == 0:
            return False
        return True

    # @Profiler("to_rpc_task")
    def to_rpc_task(self):
        input = self.input
        output = self.output

        def tensor(t: torch.Tensor):
            if t is None:
                return
            t = t.contiguous()
            npt = t.numpy(force=True)
            return messages_pb2.LocalSharedTensor(
                byte=npt.tobytes(),
                shape=t.shape,
                dtype=npt.dtype.name,
            )

        rpc_input = None
        if input is not None:
            rpc_input = messages_pb2.Data(
                index=input.get("index"),
                target=input.get("target"),
                question=input.get("question"),
                passages=(
                    json.dumps(input["passages"])
                    if Task.check_non_none(input.get("passages"))
                    else None
                ),
                scores=tensor(input.get("scores")),
                contexts=tensor(input.get("contexts")),
                context_masks=tensor(input.get("context_masks")),
                dep_idx=tensor(input.get("dep_idx")),
                tokens=tensor(input.get("tokens")),
            )
        rpc_output = None
        if output is not None:
            rpc_output = messages_pb2.Data(
                contexts=tensor(output.get("contexts")),
                context_masks=tensor(output.get("context_masks")),
                tokens=tensor(output.get("tokens")),
                text_ans=output.get("text_ans"),
            )
        rpc_task = messages_pb2.Task(input=rpc_input, output=rpc_output)
        return rpc_task

    # @Profiler("from_rpc_task")
    def from_rpc_task(rpc_task: messages_pb2.Task):
        task = Task()
        input = rpc_task.input
        output = rpc_task.output

        def tensor(t: messages_pb2.LocalSharedTensor):
            if t is None or len(t.byte) == 0:
                return
            nparray = np.frombuffer(t.byte, dtype=np.dtype(t.dtype)).reshape(t.shape)
            return torch.from_numpy(nparray)

        if input is not None:
            task.input = {
                "index": input.index,
                "target": input.target,
                "question": input.question,
                "passages": (
                    json.loads(input.passages)
                    if Task.check_non_none(input.passages)
                    else None
                ),
                "scores": tensor(input.scores),
                "contexts": tensor(input.contexts),
                "context_masks": tensor(input.context_masks),
                "dep_idx": tensor(input.dep_idx),
            }
        if output is not None:
            task.output = {
                # "scores": tensor(output.scores),
                "contexts": tensor(output.contexts),
                "context_masks": tensor(output.context_masks),
                "tokens": tensor(output.tokens),
                "text_ans": output.text_ans,
            }
        return task


class TaskQueue(List[Task]):
    def total_waiting_time(self):
        now = time.time()
        total = 0
        for task in self:
            total += now - task.request.arrive_time
        return total

    def pop_encoder_tasks(self, passages_size):
        if len(self) == 0:
            return []
        now = time.time()

        def sort_key(x):
            waiting = now - x.request.arrive_time
            n_passages = len(x.input["passages"])
            value = waiting / n_passages
            return value

        sorted_tasks = sorted(self, key=sort_key, reverse=True)
        curr_size = 0
        ret_tasks = []
        for x in sorted_tasks:
            n_passages = len(x.input["passages"])
            if curr_size + n_passages > passages_size:
                break
            curr_size += n_passages
            ret_tasks.append(x)
            self.remove(x)
        if len(ret_tasks) == 0:
            x = sorted_tasks[0]
            ret_tasks.append(x)
            self.remove(x)

        print(f"passages per task: {[len(x.input['passages']) for x in ret_tasks]} ")

        return ret_tasks

    def pop_earliest_tasks(self, size, need_dependency=False):
        out = []
        tasks = self
        if need_dependency:
            tasks[:] = [x for x in tasks if x.dep is None or x.dep.resolve()]
        while len(out) < size and len(tasks) > 0:
            x = min(tasks, key=lambda x: x.request.arrive_time)
            self.remove(x)
            out.append(x)
        return out


class BatchTask:

    def __init__(self):
        self.tasks = []
        self.future: grpc.Future = None

    def add_tasks(self, tasks):
        self.tasks += tasks
        return self

    def get_env_type(self):
        return self.tasks[0].env_type

    def passage_per_task(self):
        return [len(task.input["passages"]) for task in self.tasks]

    def rpc_execute(self, stub):
        pass

    def post_process(self):
        pass


class BatchEncoderTask(BatchTask):

    def rpc_execute(self, stub: EncoderServiceStub):
        print(f"rpc_execute({self.get_env_type()}, Encoder)")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        self.future = stub.ExecuteBatchEncoderTask.future(request)

    def post_process(self):
        if not self.future.done():
            return
        response = self.future.result()
        ret_tasks = [Task.from_rpc_task(rpc_task) for rpc_task in response.tasks]
        for task, ret in zip(self.tasks, ret_tasks):
            task.output = ret.output
            task.is_finished = True
        return self.tasks


class BatchDecoderTask(BatchTask):

    def rpc_execute(self, stub: DecoderServiceStub):
        print(f"rpc_execute({self.get_env_type()}, Decoder)")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        self.future = stub.ExecuteBatchDecoderTask.future(request)

    def post_process(self):
        if not self.future.done():
            return
        response = self.future.result()
        ret_tasks = [Task.from_rpc_task(rpc_task) for rpc_task in response.tasks]
        for task, ret in zip(self.tasks, ret_tasks):
            task.output = ret.output
            print(
                f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
            )
            task.is_finished = True
        return self.tasks


class BatchContinueEncoderDecoderTask(BatchTask):

    def rpc_execute(self, stub: EncoderServiceStub):
        print(f"rpc_execute({self.get_env_type()}, Encoder)")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        self.future = stub.ExecuteBatchContinueEncoderDecoderTask.future(request)

    def post_process(self):
        if not self.future.done():
            return
        response = self.future.result()
        ret_tasks = [Task.from_rpc_task(rpc_task) for rpc_task in response.tasks]
        for task, ret in zip(self.tasks, ret_tasks):
            task.output = ret.output
            task.is_finished = True
            print(
                f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
            )
        return self.tasks


class BatchGenerateTask(BatchTask):
    def rpc_execute(self, stub, enable_offloading=True):
        print(f"rpc_execute({self.get_env_type()}, Generate)")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        if enable_offloading:
            passage_size = len(self.tasks[0].input["passages"])
            request.max_private_ratio = max(
                [
                    task.request.private_passage_size / passage_size
                    for task in self.tasks
                ]
            )
            print(f"request.max_private_ratio: {request.max_private_ratio}")
            self.future = stub.ExecuteBatchOffloadingGenerateTask.future(request)
        else:
            self.future = stub.ExecuteBatchGenerateTask.future(request)

    def post_process(self):
        if not self.future.done():
            return
        response = self.future.result()
        ret_tasks = [Task.from_rpc_task(rpc_task) for rpc_task in response.tasks]
        for task, ret in zip(self.tasks, ret_tasks):
            task.output = ret.output
            print(
                f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
            )
            task.is_finished = True
        return self.tasks


class EncoderDecoderSerivce(
    EncoderService, DecoderService, MetricService, GenerateService
):
    def __init__(self, cfg, type):
        self.pad_token_id = 0

        tee_port = cfg.tee_service_port
        gpu_port = cfg.gpu_service_port
        self.text_maxlength = cfg.text_maxlength
        self.answer_maxlength = cfg.answer_maxlength

        model_path = cfg.generator_model_path
        model: FiDT5 = FiDT5.from_pretrained(model_path)
        model = model.eval()
        self.encoder = model.get_encoder().encoder
        self.decoder = model.get_decoder()
        self.tokenizer: transformers.T5Tokenizer = (
            transformers.T5Tokenizer.from_pretrained(
                "models/t5-base", return_dict=False
            )
        )

        self.port = None
        self.type = type
        self.device = "cpu"
        if type == "TEE":
            self.port = tee_port
            self.device = "cpu"
        elif type == "GPU":
            self.port = gpu_port
            self.device = "cuda"
        elif type == "NATIVE":
            self.port = 8082
            self.device = "cpu"
        self.model = model.to(self.device)

        if type == "OFFLAODING":
            self.port = 8083
            self.model = SecureRAG(fidt5=self.model)
            self.collator = SecureRAG4T5Collator(
                tokenizer=self.tokenizer,
                text_maxlength=self.text_maxlength,
                answer_maxlength=self.answer_maxlength,
                private_passage_ratio=0,  # dynamic in runtime
            )

    def passage_per_task(self, batch_input):
        return [len(data["passages"]) for data in batch_input]

    def _tokenizer(self, batch):
        passages = []
        for example in batch:
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
            passages.append(
                [
                    f.format(example["question"], t["title"], t["text"])
                    for t in example["passages"]
                ]
            )
        passage_ids, passage_masks = tokenizer_encode_batch2(
            passages, self.tokenizer, self.text_maxlength
        )
        return BatchData(
            passage_ids=passage_ids,
            passage_masks=passage_masks,
        )

    @torch.inference_mode()
    @Profiler("ExecuteBatchEncoderTask")
    def ExecuteBatchEncoderTask(self, request, context):
        tasks = request.tasks
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        batch = self._tokenizer(batch_input)
        (
            context_ids,
            context_masks,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
        )
        is_cuda = True if self.type == "GPU" else False
        with Profiler(f"{self.type}_ENCODER", is_cuda=is_cuda):
            output = self.encoder(
                input_ids=context_ids.to(self.device),
                attention_mask=context_masks.to(self.device),
                return_dict=True,
            )
        encoder_contexts = output["last_hidden_state"].to("cpu")
        passages_dim = self.passage_per_task(batch_input)
        ret_tasks = []
        curr = 0
        for dim in passages_dim:
            tmp = Task()
            tmp.output = {
                "contexts": encoder_contexts[curr : curr + dim],
                "context_masks": context_masks[curr : curr + dim],
            }
            curr += dim
            ret_tasks.append(tmp)

        add_metric("finished_encoder_task", 1)
        show_metric()

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    @torch.inference_mode()
    @Profiler("ExecuteBatchDecoderTask")
    def ExecuteBatchDecoderTask(self, request, context):
        tasks = request.tasks
        batch_size = len(tasks)
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        contexts = [input["contexts"] for input in batch_input]
        masks = [input["context_masks"] for input in batch_input]
        maxlen = max([context.size(0) for context in contexts])
        contexts = torch.stack(
            [
                pad(x, (0, 0, 0, maxlen - x.size(0)), value=self.pad_token_id)
                for x in contexts
            ]
        )
        masks = torch.stack(
            [pad(x, (0, maxlen - x.size(0)), value=False) for x in masks]
        )
        encoder_outputs = ModelOutput()
        encoder_outputs["last_hidden_state"] = contexts.to(self.device)
        is_cuda = True if self.type == "GPU" else False
        with Profiler(f"{self.type}_DECODER", is_cuda=is_cuda):
            ans = self.model.generate_without_encoder(
                input_ids=torch.empty(batch_size, 1).to(
                    self.device
                ),  # useless input_ids
                encoder_outputs=encoder_outputs,
                attention_mask=masks.to(self.device),
            )
        ans = ans.to("cpu")
        batch_text_ans = self.tokenizer.batch_decode(ans, skip_special_tokens=True)
        print(batch_text_ans)
        ret_tasks = []
        for idx, tokens in enumerate(ans):
            text_ans = batch_text_ans[idx]
            tmp = Task()
            tmp.output = {
                "tokens": tokens,
                "text_ans": text_ans,
            }
            ret_tasks.append(tmp)

        add_metric("finished_decoder_task", 1)
        show_metric()

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    @torch.inference_mode()
    @Profiler("ExecuteBatchContinueEncoderDecoderTask")
    def ExecuteBatchContinueEncoderDecoderTask(self, request, context):
        tasks = request.tasks
        batch_size = len(tasks)
        batch_input = [Task.from_rpc_task(task).input for task in tasks]

        # 1️⃣ 批量 tokenizer
        batch = self._tokenizer(batch_input)
        context_ids, context_masks = batch.passage_ids.to(
            self.device
        ), batch.passage_masks.to(self.device)

        # 2️⃣ encoder 前向，全部在 GPU
        is_cuda = self.type == "GPU"
        with Profiler(f"{self.type}_ENCODER", is_cuda=is_cuda):
            output = self.encoder(
                input_ids=context_ids,
                attention_mask=context_masks,
                return_dict=True,
            )
        encoder_contexts = output["last_hidden_state"]  # GPU tensor，保留在 GPU

        # 3️⃣ 计算每个 task 的长度，用 chunk 代替 Python 循环切片
        passages_dim = self.passage_per_task(batch_input)
        cum_dims = [0] + list(torch.cumsum(torch.tensor(passages_dim), dim=0).tolist())
        contexts_list = [
            encoder_contexts[cum_dims[i] : cum_dims[i + 1]]
            for i in range(len(passages_dim))
        ]
        masks_list = [
            context_masks[cum_dims[i] : cum_dims[i + 1]]
            for i in range(len(passages_dim))
        ]

        # 4️⃣ 创建 decoder tasks
        ret_tasks = []
        for idx, (c, m) in enumerate(zip(contexts_list, masks_list)):
            tmp = Task()
            dep_idx = batch_input[idx]["dep_idx"]
            if dep_idx is not None:
                dep_idx = dep_idx.to(c.device)
                tmp.output = {"contexts": c[dep_idx], "context_masks": m[dep_idx]}
                add_metric("rpc_contexts_size", m.size(0))
            ret_tasks.append(tmp)

        # fusion
        contexts_list = [
            c.view(c.size(0) * c.size(1), c.size(2)) for c in contexts_list
        ]
        masks_list = [m.view(m.size(0) * m.size(1)) for m in masks_list]
        # 5️⃣ GPU 上 padding
        maxlen = max([c.size(0) for c in contexts_list])
        contexts_padded = torch.stack(
            [
                torch.nn.functional.pad(
                    c, (0, 0, 0, maxlen - c.size(0)), value=self.pad_token_id
                )
                for c in contexts_list
            ]
        )
        masks_padded = torch.stack(
            [
                torch.nn.functional.pad(m, (0, maxlen - m.size(0)), value=False)
                for m in masks_list
            ]
        )

        # 6️⃣ decoder 前向
        encoder_outputs = ModelOutput()
        encoder_outputs["last_hidden_state"] = contexts_padded

        with Profiler(f"{self.type}_DECODER", is_cuda=is_cuda):
            ans = self.model.generate_without_encoder(
                input_ids=torch.empty(
                    batch_size, 1, device=self.device
                ),  # dummy input_ids
                encoder_outputs=encoder_outputs,
                attention_mask=masks_padded,
            )
        ans = ans.to("cpu")  # batch_decode 在 CPU

        # 7️⃣ tokenizer decode（可以多线程加速）
        batch_text_ans = self.tokenizer.batch_decode(ans, skip_special_tokens=True)

        for tmp, tokens, text_ans in zip(ret_tasks, ans, batch_text_ans):
            if tmp.output is None:
                tmp.output = {}
            tmp.output["tokens"] = tokens
            tmp.output["text_ans"] = text_ans

        add_metric("finished_continue_encoder_decoder_task", 1)
        show_metric()

        # 8️⃣ 清理 GPU 内存
        del encoder_outputs, contexts_padded, masks_padded, ans
        torch.cuda.empty_cache()

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    @torch.inference_mode()
    @Profiler("ExecuteBatchContinueEncoderDecoderTask")
    def ExecuteBatchContinueEncoderDecoderTaskOLD(self, request, context):
        tasks = request.tasks
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        batch = self._tokenizer(batch_input)
        (
            context_ids,
            context_masks,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
        )
        is_cuda = True if self.type == "GPU" else False
        with Profiler(f"{self.type}_ENCODER", is_cuda=is_cuda):
            output = self.encoder(
                input_ids=context_ids.to(self.device),
                attention_mask=context_masks.to(self.device),
                return_dict=True,
            )
        encoder_contexts = output["last_hidden_state"].to("cpu")
        passages_dim = self.passage_per_task(batch_input)
        ret_tasks = []
        curr = 0
        for dim in passages_dim:
            tmp = Task()
            tmp.output = {
                "contexts": encoder_contexts[curr : curr + dim],
                "context_masks": context_masks[curr : curr + dim],
            }
            curr += dim
            ret_tasks.append(tmp)
        ret_tasks = [Task.create_decoder_task(pre, new_task=False) for pre in ret_tasks]
        batch_size = len(tasks)
        batch_input = [task.input for task in ret_tasks]
        contexts = [input["contexts"] for input in batch_input]
        masks = [input["context_masks"] for input in batch_input]
        maxlen = max([context.size(0) for context in contexts])
        contexts = torch.stack(
            [
                pad(x, (0, 0, 0, maxlen - x.size(0)), value=self.pad_token_id)
                for x in contexts
            ]
        )
        masks = torch.stack(
            [pad(x, (0, maxlen - x.size(0)), value=False) for x in masks]
        )
        encoder_outputs = ModelOutput()
        encoder_outputs["last_hidden_state"] = contexts.to(self.device)
        is_cuda = True if self.type == "GPU" else False
        with Profiler(f"{self.type}_DECODER", is_cuda=is_cuda):
            ans = self.model.generate_without_encoder(
                input_ids=torch.empty(batch_size, 1).to(
                    self.device
                ),  # useless input_ids
                encoder_outputs=encoder_outputs,
                attention_mask=masks.to(self.device),
            )
        ans = ans.to("cpu")
        batch_text_ans = self.tokenizer.batch_decode(ans, skip_special_tokens=True)
        print(batch_text_ans)
        for idx, tokens in enumerate(ans):
            text_ans = batch_text_ans[idx]
            tmp = ret_tasks[idx]
            if tmp.output is None:
                tmp.output = {}
            tmp.output["tokens"] = tokens
            tmp.output["text_ans"] = text_ans

        add_metric("finished_continue_encoder_decoder_task", 1)
        show_metric()

        del encoder_outputs, contexts, masks, ans
        torch.cuda.empty_cache()

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    @torch.inference_mode()
    @Profiler("ExecuteBatchOffloadingGenerateTask")
    def ExecuteBatchOffloadingGenerateTask(self, request, context):
        tasks = request.tasks
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        bsz = len(batch_input)
        batch = self.collator(batch_input, private_ratio=request.max_private_ratio)
        (
            context_ids,
            context_masks,  # bsz * docs * dim
            private_context_ids,
            private_context_masks,  # bsz * docs_p * dim
            scores,  # bsz * docs
            private_scores,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
            batch.private_passage_ids,
            batch.private_passage_masks,
            batch.scores,
            batch.private_scores,
        )
        context_ids = context_ids.view(bsz, -1, context_ids.size(-1))
        context_masks = context_masks.view(bsz, -1, context_masks.size(-1))
        private_context_ids = private_context_ids.view(
            bsz, -1, private_context_ids.size(-1)
        )
        private_context_masks = private_context_masks.view(
            bsz, -1, private_context_masks.size(-1)
        )
        output = self.model.generate(
            context_ids=context_ids,
            context_ids_private=private_context_ids,
            attention_mask=context_masks,
            attention_mask_private=private_context_masks,
            doc_scores=scores,
            doc_scores_private=private_scores,
            max_length=self.answer_maxlength,
        )
        ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(ans)
        ret_tasks = []
        for idx, _ in enumerate(ans):
            text_ans = ans[idx]
            tmp = Task()
            tmp.output = {
                "text_ans": text_ans,
            }
            ret_tasks.append(tmp)

        add_metric("finished_generate_task", 1)
        show_metric()

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    @torch.inference_mode()
    @Profiler("ExecuteBatchGenerateTask")
    def ExecuteBatchGenerateTask(self, request, context):
        tasks = request.tasks
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        bsz = len(batch_input)
        batch = self._tokenizer(batch_input)
        (
            context_ids,
            context_masks,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
        )
        context_ids = context_ids.view(bsz, -1, context_ids.size(-1))
        context_masks = context_masks.view(bsz, -1, context_masks.size(-1))
        output = self.model.generate(
            input_ids=context_ids.to(self.device),
            attention_mask=context_masks.to(self.device),
            max_length=self.answer_maxlength,
        )
        ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(ans)
        ret_tasks = []
        for idx, _ in enumerate(ans):
            text_ans = ans[idx]
            tmp = Task()
            tmp.output = {
                "text_ans": text_ans,
            }
            ret_tasks.append(tmp)

        add_metric("finished_generate_task", 1)
        show_metric()

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    def ClearMetric(self, request, context):
        print(f"{self.type} service: delete metric")
        delete_metric()
        return empty_pb2.Empty()

    def start_service(self, worker_num=1):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=worker_num),
            options=[
                ("grpc.max_message_length", -1),
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )
        add_EncoderServiceServicer_to_server(self, server)
        add_DecoderServiceServicer_to_server(self, server)
        add_GenerateServiceServicer_to_server(self, server)
        add_MetricServiceServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        print(f"{self.type} service is running...")
        server.wait_for_termination()
        server.wait_for_termination()


class LocalEncoderDecoderService(EncoderDecoderSerivce):
    def __init__(self, cfg, type):
        super().__init__(cfg, type)
        # self.gpu_model = deepcopy(self.model).to("cuda")
        # self.gpu_encoder = self.gpu_model.get_encoder().encoder

    @torch.inference_mode()
    def ExecuteBatchGenerateTask(self, tasks, device="cpu"):
        batch_input = [task.input for task in tasks]
        bsz = len(batch_input)
        batch = self._tokenizer(batch_input)
        (
            context_ids,
            context_masks,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
        )
        context_ids = context_ids.view(bsz, -1, context_ids.size(-1))
        context_masks = context_masks.view(bsz, -1, context_masks.size(-1))
        if device == "cpu":
            model = self.model
        elif device == "cuda":
            model = self.gpu_model
        (output, encoder_contexts) = model.generate(
            input_ids=context_ids.to(device),
            attention_mask=context_masks.to(device),
            max_length=self.answer_maxlength,
            return_encoder_outputs=True,
        )
        ans = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(ans)

        passages_dim = self.passage_per_task(batch_input)
        curr = 0
        for idx, dim in enumerate(passages_dim):
            text_ans = ans[idx]
            tmp = tasks[idx]
            tmp.output = {
                "text_ans": text_ans,
                "contexts": encoder_contexts[curr : curr + dim],
                "context_masks": context_masks[curr : curr + dim],
            }
            curr += dim

    @torch.inference_mode()
    def ExecuteBatchEncoderTask(self, tasks, device="cpu"):
        batch_input = [task.input for task in tasks]
        with Profiler("tokenizer_tee"):
            batch = self._tokenizer(batch_input)
        (
            context_ids,
            context_masks,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
        )
        if device == "cpu":
            encoder = self.encoder
        elif device == "cuda":
            encoder = self.gpu_encoder
        is_cuda = True if device == "cuda" else False
        with Profiler(f"{device}_ENCODER", is_cuda=is_cuda):
            output = encoder(
                input_ids=context_ids.to(device),
                attention_mask=context_masks.to(device),
                return_dict=True,
            )
        encoder_contexts = output["last_hidden_state"].to("cpu")
        passages_dim = self.passage_per_task(batch_input)
        curr = 0
        for idx, dim in enumerate(passages_dim):
            tmp = tasks[idx]
            tmp.output = {
                "contexts": encoder_contexts[curr : curr + dim],
                "context_masks": context_masks[curr : curr + dim],
            }
            curr += dim

    @torch.inference_mode()
    def ExecuteBatchDecoderTask(self, tasks, device="cpu"):
        batch_size = len(tasks)
        batch_input = [task.input for task in tasks]
        contexts = [input["contexts"] for input in batch_input]
        masks = [input["context_masks"] for input in batch_input]
        maxlen = max([context.size(0) for context in contexts])
        contexts = torch.stack(
            [
                pad(x, (0, 0, 0, maxlen - x.size(0)), value=self.pad_token_id)
                for x in contexts
            ]
        )
        masks = torch.stack(
            [pad(x, (0, maxlen - x.size(0)), value=False) for x in masks]
        )
        encoder_outputs = ModelOutput()
        encoder_outputs["last_hidden_state"] = contexts.to(device)

        if device == "cpu":
            model = self.model
        elif device == "cuda":
            model = self.gpu_model

        is_cuda = True if device == "cuda" else False
        with Profiler(f"{device}_DECODER", is_cuda=is_cuda):
            ans = model.generate_without_encoder(
                input_ids=torch.empty(batch_size, 1).to(device),  # useless input_ids
                encoder_outputs=encoder_outputs,
                attention_mask=masks.to(device),
            )
        ans = ans.to("cpu")
        batch_text_ans = self.tokenizer.batch_decode(ans, skip_special_tokens=True)
        print(batch_text_ans)
        for idx, tokens in enumerate(ans):
            text_ans = batch_text_ans[idx]
            tmp = tasks[idx]
            if tmp.output is None:
                tmp.output = {}
            tmp.output["tokens"] = tokens
            tmp.output["text_ans"] = text_ans
