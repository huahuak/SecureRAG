import json
import time
from collections import defaultdict
from concurrent import futures
from typing import List, overload

import grpc
import numpy as np
import torch
import transformers
from pyexpat.errors import messages
from torch.nn.functional import pad
from transformers.file_utils import ModelOutput

from securerag.config import Config
from securerag.data import BatchData, tokenizer_encode_batch
from securerag.models.fid import FiDT5
from securerag.profiler import Profiler
from securerag.rpc import messages_pb2
from securerag.rpc.messages_pb2_grpc import (
    DecoderService,
    DecoderServiceStub,
    EncoderService,
    EncoderServiceStub,
    add_DecoderServiceServicer_to_server,
    add_EncoderServiceServicer_to_server,
)
from securerag.scheduler.dependency import FusionAggregate
from securerag.scheduler.requests import Request
from securerag.utils import show_metric


class Task:
    def __init__(self):
        self.request = None
        self.env_type = "TEE"  # default env
        self.input = None
        self.output = None
        self.dep = None
        self.is_finished = False

    @staticmethod
    def create_task_from_request(req: Request):
        data = req.input_data
        private_passage_size = req.private_passage_size
        print(f"private_passage_size: {private_passage_size}")

        public_data = {
            "index": data["index"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"][private_passage_size:],
            "scores": data["scores"][private_passage_size:],
        }

        private_data = {
            "index": data["index"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"][:private_passage_size],
            "scores": data["scores"][:private_passage_size],
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

        return (public_task, private_task)

    def create_decoder_task(pre_task):
        task = Task()
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
                question=input.get("question"),
                passages=(
                    json.dumps(input["passages"])
                    if Task.check_non_none(input.get("passages"))
                    else None
                ),
                scores=tensor(input.get("scores")),
                contexts=tensor(input.get("contexts")),
                context_masks=tensor(input.get("context_masks")),
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
                "question": input.question,
                "passages": (
                    json.loads(input.passages)
                    if Task.check_non_none(input.passages)
                    else None
                ),
                "scores": tensor(input.scores),
                "contexts": tensor(input.contexts),
                "context_masks": tensor(input.context_masks),
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

    @Profiler("rpc_execute_encoder")
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

    @Profiler("rpc_execute_decoder")
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


class EncoderDecoderSerivce(EncoderService, DecoderService):
    def __init__(self, cfg, type):
        self.pad_token_id = 0

        tee_port = cfg.tee_service_port
        gpu_port = cfg.gpu_service_port
        self.text_maxlength = cfg.text_maxlength
        self.answer_maxlength = cfg.answer_maxlength

        model_path = cfg.generator_model_path
        model: FiDT5 = FiDT5.from_pretrained(model_path)
        model = model.eval()
        torch.no_grad()
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
        self.model = model.to(self.device)

        self.runtime_status = defaultdict(int)

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
        passage_ids, passage_masks = tokenizer_encode_batch(
            passages, self.tokenizer, self.text_maxlength
        )
        return BatchData(
            passage_ids=passage_ids,
            passage_masks=passage_masks,
        )

    @torch.inference_mode()
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

        show_metric()
        self.runtime_status["ExecuteBatchEncoderTask"] += 1
        print(f"{self.type} runtime status: {self.runtime_status}")

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    @torch.inference_mode()
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

        show_metric()
        self.runtime_status["ExecuteBatchDecoderTask"] += 1
        print(f"{self.type} runtime status: {self.runtime_status}")

        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

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
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        print(f"{self.type} service is running...")
        server.wait_for_termination()
