import os
import queue
import random
import sys
import time
from concurrent import futures
from typing import List

import grpc
from google.protobuf import empty_pb2
from torch import signal

from securerag import data
from securerag.rpc import messages_pb2
from securerag.rpc.messages_pb2_grpc import (
    MetricService,
    add_GenerateServiceServicer_to_server,
)
from securerag.scheduler.tasks import (
    GenerateService,
    Task,
    add_MetricServiceServicer_to_server,
)
from securerag.utils import add_metric, delete_metric


class Request:
    def __init__(self):
        self.arrive_time = None
        self.finish_time = None
        self.first_token_time = None
        self.token_between_time = []
        self.input_data = None
        self.private_passage_size = 0
        self.private_ans = None
        self.public_ans = None
        self.final_ans = None


class RequestSource:
    def arrive_requests(self) -> List[Request]:
        return []


class RpcRequestSource(RequestSource, GenerateService, MetricService):
    def __init__(self, port):
        self.max_queue_size = 1
        self.queue = queue.Queue(self.max_queue_size)
        self.port = port

    def arrive_requests(self):
        reqs = []
        while not self.queue.empty():
            reqs.append(self.queue.get())
        return reqs

    def ExecuteBatchContinueEncoderDecoderTask(self, request, context):
        tasks = request.tasks
        batch_tasks = [Task.from_rpc_task(task) for task in tasks]
        for idx, task in enumerate(batch_tasks):
            req = Request()
            req.input_data = task.input
            req.private_passage_size = request.tasks[idx].private_passage_size
            print(
                f"receive request..., private_passage_size is {req.private_passage_size}"
            )
            req.arrive_time = time.time()
            self.queue.put(req)
        return messages_pb2.Response(credit=self.max_queue_size - self.queue.qsize())

    def ClearMetric(self, request, context):
        print(f"TEE Instance service: delete metric")
        delete_metric()
        add_metric("start_time", time.time())
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
        add_GenerateServiceServicer_to_server(self, server)
        add_MetricServiceServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        print(f"TEE Instance service is running...")
        server.wait_for_termination()


class LocalRequestSource(RequestSource):
    def __init__(self):
        random.seed(2026)

        self.dataset = None
        self.curr = 0
        self.lasttime = time.time()
        self.request_per_second = 64

    def registry_source(self, path, config):
        datas = data.load(path=path, size=config.load_size)
        self.dataset = data.Dataset(data=datas, n_context=config.n_context)

    def arrive_requests(self) -> List[Request]:
        now = time.time()
        interval = now - self.lasttime

        request_size = int(interval * self.request_per_second)
        if request_size > 0:
            self.lasttime = time.time()
        data_size = len(self.dataset)

        output = []
        while self.curr < data_size and request_size > 0:
            req = Request()
            req.input_data = self.dataset[self.curr]
            req = self.request_load_post_process(req)
            req.arrive_time = time.time()
            output.append(req)
            request_size -= 1
            self.curr += 1

        return output

    def request_load_post_process(self, req):  # -> Any:
        n_passages = len(req.input_data["passages"])
        siz = (
            min(random.randint(5, 10), n_passages)
            if random.random() < 0.8
            else min(random.randint(20, 50), n_passages)
        )
        req.private_passage_size = (
            random.randint(0, int(siz * 0.6))
            if random.random() < 0.9
            else random.randint(int(siz * 0.6), siz)
        )
        req.input_data["passages"] = req.input_data["passages"][:siz]
        req.input_data["scores"] = req.input_data["scores"][:siz]

        return req
