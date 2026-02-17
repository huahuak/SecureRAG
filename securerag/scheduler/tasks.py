import json
import time
from concurrent import futures
from typing import List, overload

import grpc
import numpy as np
import torch
import transformers
from pyexpat.errors import messages
from torch.nn.functional import pad

from securerag.config import Config
from securerag.data import BatchData, tokenizer_encode_batch
from securerag.models.fid import FiDT5
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
            public_task.env_type = "TEE"
            private_task.input = private_data

        if public_task and private_task:
            private_task.dep = FusionAggregate(public_task, private_task)

        return (public_task, private_task)

    def create_decoder_task(pre_task):
        task = Task()
        task.request = pre_task.request
        task.env_type = pre_task.env_type
        task.input = {
            # extract input
            "scores": pre_task.input["scores"],
            # extract output
            "contexts": pre_task.output["contexts"],
            "context_masks": pre_task.output["context_masks"],
        }
        return task

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
                question=input["question"] if "question" in input else None,
                passages=json.dumps(input["passages"]) if "passages" in input else None,
                scores=tensor(input["scores"]) if "scores" in input else None,
                contexts=tensor(input["contexts"]) if "contexts" in input else None,
                context_masks=(
                    tensor(input["context_masks"]) if "context_masks" in input else None
                ),
                tokens=tensor(input["tokens"]) if "tokens" in input else None,
            )
        rpc_output = None
        if output is not None:
            rpc_output = messages_pb2.Data(
                contexts=tensor(output["contexts"]) if "contexts" in output else None,
                context_masks=(
                    tensor(output["context_masks"])
                    if "context_masks" in output
                    else None
                ),
                tokens=tensor(output["tokens"]) if "tokens" in output else None,
            )
        rpc_task = messages_pb2.Task(input=rpc_input, output=rpc_output)
        return rpc_task

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
                "question": input.question if input.question else None,
                "passages": json.loads(input.passages) if input.passages else None,
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
            }
        return task


class TaskQueue(List[Task]):
    def total_waiting_time(self):
        now = time.time()
        total = 0
        for task in self:
            total += now - task.request.arrive_time
        return total

    def pop_shortest_tasks(self, size):
        out = []
        while len(out) < size and len(self) > 0:
            idx, _ = min(enumerate(self), key=lambda x: x[1].request.arrive_time)
            task = self.pop(idx)
            out.append(task)
        return out


class BatchTask:

    def __init__(self):
        self.future: grpc.Future = None

    def add_tasks(self, tasks):
        self.tasks = tasks
        return self

    def passage_per_task(self):
        return [len(task.input["passages"]) for task in self.tasks]

    def rpc_execute(self, stub):
        pass

    def post_process(self):
        pass


class BatchEncoderTask(BatchTask):
    def rpc_execute(self, stub: EncoderServiceStub):
        print("encoder batch task send rpc_execute...")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        self.future = stub.ExecuteBatchEncoderTask.future(request)
        self.future.result()

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
        print("decoder batch task send rpc_execute...")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        self.future = stub.ExecuteBatchDecoderTask.future(request)
        self.future.result()


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
        output = self.encoder(
            input_ids=context_ids.to(self.device),
            attention_mask=context_masks.to(self.device),
            return_dict=True,
        )
        encoder_contexts = output["last_hidden_state"]
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
        return messages_pb2.Response(
            tasks=[Task.to_rpc_task(ret_task) for ret_task in ret_tasks]
        )

    def ExecuteBatchDecoderTask(self, request, context):
        tasks = request.tasks
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        contexts = [input["contexts"] for input in batch_input]
        masks = [input["context_masks"] for input in batch_input]
        maxlen = max([context.size(0) for context in contexts])
        contexts = torch.Stack(
            [pad(x, (0, maxlen - x.size(0)), value=self.pad_token_id) for x in contexts]
        )
        masks = torch.Stack(
            [pad(x, (0, maxlen - x.size(0)), value=False) for x in masks]
        )
        ans = self.model.generate_without_encoder(
            encoder_outputs=contexts, attention_mask=masks
        )
        print(ans.shape)

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
        server.wait_for_termination()
        server.wait_for_termination()
