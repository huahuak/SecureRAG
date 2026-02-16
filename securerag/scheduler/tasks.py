import json
import time
from concurrent import futures
from typing import List, overload

import grpc
import numpy as np
import torch
import transformers
from pyexpat.errors import messages

from securerag.config import Config
from securerag.data import BatchData, tokenizer_encode_batch
from securerag.models.fid import FiDT5
from securerag.rpc import messages_pb2
from securerag.rpc.messages_pb2_grpc import (
    DecoderService,
    EncoderService,
    EncoderServiceStub,
    add_DecoderServiceServicer_to_server,
    add_EncoderServiceServicer_to_server,
)
from securerag.scheduler.requests import Request


class Task:
    def __init__(self):
        self.request = None
        self.env_type = "TEE"  # default env
        self.input = None
        self.output = None

    @staticmethod
    def create_task_from_request(req: Request):
        data = req.input_data
        private_passage_size = req.private_passage_size

        public_data = {
            "index": data["question"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"][private_passage_size:],
            "scores": data["scores"][private_passage_size:],
        }

        private_data = {
            "index": data["question"],
            "question": data["question"],
            "target": data["target"],
            "passages": data["passages"][:private_passage_size],
            "scores": data["scores"][:private_passage_size],
        }

        public_task = Task()
        public_task.request = req
        public_task.input = public_data

        private_task = Task()
        private_task.request = req
        private_task.input = private_data

        return (public_task, private_task)

    def to_rpc_task(self):
        input = self.input
        output = self.output

        def tensor(t: torch.Tensor):
            if t is None:
                return
            t.contiguous()
            return messages_pb2.LocalSharedTensor(byte=t.numpy().tobytes(), shape=t.shape)

        rpc_input = messages_pb2.Data(
            question=input["question"],
            passages=json.dumps(input["passages"]),
            scores=tensor(input["scores"]),
        )
        rpc_output = None
        if output is not None:
            rpc_output = messages_pb2.Data(
                contexts=tensor(output.contexts),
                tokens=tensor(output.tokens),
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
            nparray = np.frombuffer(t.byte, dtype=np.float32).reshape(t.shape)
            return torch.from_numpy(nparray)

        task.input = {
            "question": input.question,
            "passages": json.loads(input.passages),
            "scores": tensor(input.scores),
        }
        task.output = {
            # "scores": tensor(output.scores),
            "contexts": tensor(output.contexts),
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
    def add_tasks(self, tasks):
        self.tasks = tasks
        return self

    def rpc_execute(self, stub):
        pass


class BatchEncoderTask(BatchTask, EncoderService):
    def __init__(self):
        self.port = 8080
        pass

    def rpc_execute(self, stub: EncoderServiceStub):
        print("rpc_execute...")
        rpc_tasks = []
        for task in self.tasks:
            rpc_tasks.append(task.to_rpc_task())
        request = messages_pb2.Request(tasks=rpc_tasks)
        # stub.ExecuteBatchEncoderTask.future(request)
        stub.ExecuteBatchEncoderTask(request)

    def _tokenizer(self, batch):
        scores = torch.stack([ex["scores"] for ex in batch])
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
            for t in example["passages"]:
                passages.append(f.format(example["question"], t["title"], t["text"]))
        passage_ids, passage_masks = tokenizer_encode_batch(
            passages, self.tokenizer, self.text_maxlength
        )
        return BatchData(
            passage_ids=passage_ids,
            passage_masks=passage_masks,
            scores=scores,
        )

    def ExecuteBatchEncoderTask(self, request, context):
        tasks = request.tasks
        batch_input = [Task.from_rpc_task(task).input for task in tasks]
        batch = self._tokenizer(batch_input)
        (
            context_ids,
            context_masks,
            scores,
        ) = (
            batch.passage_ids,
            batch.passage_masks,
            batch.scores,
        )
        context_ids = context_ids.view(context_ids.size(0), -1)
        context_masks = context_masks.view(context_masks.size(0), -1)
        output = self.encoder(
            input_ids=context_ids.to(self.device),
            attention_mask=context_masks.to(self.device),
            max_length=50,
        )
        return messages_pb2.Response()

    def start_service(self, cfg):
        self.text_maxlength = cfg.text_maxlength

        model_path = cfg.generator_model_path
        model: FiDT5 = FiDT5.from_pretrained(model_path)
        model.to("cpu")
        model.eval()
        self.encoder = model.get_encoder()

        self.tokenizer: transformers.T5Tokenizer = (
            transformers.T5Tokenizer.from_pretrained(
                "models/t5-base", return_dict=False
            )
        )

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_EncoderServiceServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        print("Encoder service running...")
        server.wait_for_termination()


class BatchDecoderTask(BatchTask, DecoderService):
    def __init__(self):
        pass

    def __init__(self):
        pass

    def start_service(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_EncoderServiceServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        print("Decoder service running...")
        server.wait_for_termination()
