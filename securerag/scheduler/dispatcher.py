import time
from concurrent.futures import ThreadPoolExecutor, thread
from threading import Lock
from urllib import request

import grpc
import numpy as np
import psutil
import torch
from google.protobuf import empty_pb2

from securerag.data import Profiler
from securerag.rpc import messages_pb2_grpc
from securerag.scheduler.requests import RequestSource
from securerag.scheduler.tasks import (
    BatchContinueEncoderDecoderTask,
    BatchDecoderTask,
    BatchEncoderTask,
    BatchGenerateTask,
    LocalEncoderDecoderService,
    Task,
    TaskQueue,
)
from securerag.utils import (
    add_metric,
    delete_metric,
    dump_metric,
    get_metric,
    show_metric,
)


class Dispatcher:
    def __init__(self, config):
        self.load_size = config.load_size
        self.tee_service_port = config.tee_service_port
        self.gpu_service_port = config.gpu_service_port
        self.finished_size = 0

        self.request_source = None

        self.tee_encoder_task_queue = TaskQueue()
        self.gpu_encoder_task_queue = TaskQueue()
        self.tee_decoder_task_queue = TaskQueue()
        self.gpu_decoder_task_queue = TaskQueue()
        self.rpc_execute_batch_task_queue = []

        self.running = True

        self.tee_encoder_batch_size = 4
        self.tee_decoder_batch_size = 4
        self.gpu_encoder_batch_size = 6
        self.gpu_decoder_batch_size = 6

        self.TTFT_SLO = 3.5

    def registry_request_source(self, source: RequestSource):
        self.request_source = source

    def endpoint_loop_baseline(
        self, service: LocalEncoderDecoderService, enable_offloading=False
    ):
        task_queue = TaskQueue()
        batch_size = 1
        stub = None
        if enable_offloading:
            port = 8081
            channel = grpc.insecure_channel(
                f"localhost:{port}",
                options=[
                    ("grpc.max_receive_message_length", -1),
                    ("grpc.max_send_message_length", -1),
                ],
            )
            stub = messages_pb2_grpc.MetricServiceStub(channel)
            stub.ClearMetric(empty_pb2.Empty())
            stub = messages_pb2_grpc.GenerateServiceStub(channel)
        if self.request_source is None:
            print("self.request_source must be registred")
            return
        start_time = time.time()
        while self.finished_size < self.load_size:
            reqs = self.request_source.arrive_requests()
            if not enable_offloading:
                task_queue += [
                    Task.create_native_task_from_request(req) for req in reqs
                ]
            else:
                task_queue += [Task.create_task_from_request(req) for req in reqs]
            batch = task_queue[:batch_size]
            for task in batch:
                task_queue.remove(task)
            if len(batch) == 0:
                continue
            if enable_offloading:
                gpu_batch = [x[0] for x in batch if x[0] is not None]
                batch = [x[1] for x in batch if x[1] is not None]
            if enable_offloading and len(gpu_batch) > 0:
                gpu_batch_task = BatchContinueEncoderDecoderTask().add_tasks(gpu_batch)
                gpu_batch_task.rpc_execute(stub)
            if enable_offloading and (len(batch) == 0):
                continue
            service.ExecuteBatchEncoderTask(batch)
            now = time.time()
            for task in batch:
                task.is_finished = True
                arrive_time = task.request.arrive_time
                finish_time = task.request.finish_time = time.time()
                add_metric("LATENCY_ENCODER_TEE", time.time() - arrive_time)
                add_metric("finished_encoder_tee", 1)
                arrive_time = task.request.arrive_time
                task.request.first_token_time = now
            while not all(task.check_dep() for task in batch):
                if enable_offloading and len(gpu_batch) > 0:
                    gpu_batch_task.future.result()
                    tasks = gpu_batch_task.post_process()
                    if tasks is None:
                        continue
                    for task in tasks:
                        arrive_time = task.request.arrive_time
                        finish_time = task.request.finish_time = time.time()
                        task.is_finished = True
                        print(
                            f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
                        )
                        add_metric("LATENCY_GPU", finish_time - arrive_time)
                        add_metric("finished_request_gpu", 1)
            batch = [Task.create_decoder_task(task, new_task=False) for task in batch]
            now = time.time()
            for task in batch:
                ttft_time = task.request.first_token_time
                add_metric("SYNC_TEE", now - ttft_time)
                arrive_time = task.request.arrive_time
                ttft = task.request.first_token_time = now - arrive_time
                add_metric("TTFT_TEE", ttft)
                if ttft < self.TTFT_SLO:
                    add_metric("TTFT_SLO_TEE", 1)
            service.ExecuteBatchDecoderTask(batch)
            for task in batch:
                arrive_time = task.request.arrive_time
                finish_time = task.request.finish_time = time.time()
                tpot_sum = finish_time - arrive_time - task.request.first_token_time
                task.is_finished = True
                print(
                    f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
                )
                add_metric("LATENCY_TEE", finish_time - arrive_time)
                add_metric("TPOT_TEE", tpot_sum / len(task.output.get("tokens")))
                add_metric("TOKENS_TEE", len(task.output.get("tokens")))
                add_metric("finished_request_tee", 1)
                add_metric("FINISH_TIME_TEE", time.time() - start_time)
                if len(get_metric("finished_request_tee")) == self.load_size:
                    type = "offloading" if enable_offloading else "native"
                    name = f"tmp/{type}_request{self.request_source.request_per_second}_threshold07.json"
                    dump_metric(name)
                    return
            show_metric()

    def endpoint_loop(self, service: LocalEncoderDecoderService):
        if self.request_source is None:
            print("self.request_source must be registred")
            return
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.service = service

        port = 8081
        channel = grpc.insecure_channel(
            f"localhost:{port}",
            options=[
                ("grpc.max_receive_message_length", -1),
                ("grpc.max_send_message_length", -1),
            ],
        )
        stub = messages_pb2_grpc.MetricServiceStub(channel)
        stub.ClearMetric(empty_pb2.Empty())
        stub = messages_pb2_grpc.GenerateServiceStub(channel)

        gpu_post_queue = []

        @Profiler("function_gpu")
        def gpu():
            batch = self.gpu_encoder_task_queue[: self.gpu_encoder_batch_size]
            self.gpu_encoder_task_queue[:] = self.gpu_encoder_task_queue[len(batch) :]
            if len(batch) == 0:
                return
            batch_task = BatchContinueEncoderDecoderTask().add_tasks(batch)
            batch_task.rpc_execute(stub)
            gpu_post_queue.append(batch_task)
            return

        @Profiler("function_gpu_post")
        def gpu_post():
            for batch_task in gpu_post_queue:
                tasks = batch_task.post_process()
                if tasks is None:
                    continue
                else:
                    gpu_post_queue.remove(batch_task)
                for task in tasks:
                    arrive_time = task.request.arrive_time
                    finish_time = task.request.finish_time = time.time()
                    task.is_finished = True
                    print(
                        f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
                    )
                    add_metric("LATENCY_GPU", finish_time - arrive_time)
                    add_metric("finished_request_gpu", 1)
                show_metric()

        def tee_loop():
            decoder_queue = []
            while True:

                def encoder():
                    batch_passages_size = 1
                    batch = self.tee_encoder_task_queue.pop_encoder_tasks(
                        batch_passages_size
                    )
                    if len(batch) == 0:
                        return None
                    service.ExecuteBatchEncoderTask(batch)
                    now = time.time()
                    for task in batch:
                        task.is_finished = True
                        arrive_time = task.request.arrive_time
                        add_metric("LATENCY_ENCODER_TEE", time.time() - arrive_time)
                        add_metric("finished_encoder_tee", 1)
                        arrive_time = task.request.arrive_time
                        task.request.first_token_time = now
                    show_metric()
                    return batch

                def decoder(batch):
                    batch = [
                        Task.create_decoder_task(task, new_task=False) for task in batch
                    ]
                    now = time.time()
                    for task in batch:
                        ttft_time = task.request.first_token_time
                        add_metric("SYNC_TEE", now - ttft_time)
                        arrive_time = task.request.arrive_time
                        ttft = task.request.first_token_time = now - arrive_time
                        add_metric("TTFT_TEE", ttft)
                        if ttft < self.TTFT_SLO:
                            add_metric("TTFT_SLO_TEE", 1)
                    service.ExecuteBatchDecoderTask(batch)
                    for task in batch:
                        arrive_time = task.request.arrive_time
                        finish_time = task.request.finish_time = time.time()
                        tpot_sum = (
                            finish_time - arrive_time - task.request.first_token_time
                        )
                        task.is_finished = True
                        print(
                            f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
                        )
                        add_metric("LATENCY_TEE", finish_time - arrive_time)
                        add_metric(
                            "TPOT_TEE", tpot_sum / len(task.output.get("tokens"))
                        )
                        add_metric("TOKENS_TEE", len(task.output.get("tokens")))
                        add_metric("finished_request_tee", 1)
                        add_metric("FINISH_TIME_TEE", time.time() - start_time)
                        if len(get_metric("finished_request_tee")) == self.load_size:
                            dump_metric(
                                f"tmp/sched_request{self.request_source.request_per_second}_threshold07.json"
                            )
                    show_metric()

                for batch in decoder_queue:
                    if all(task.check_dep() for task in batch):
                        decoder(batch)
                        yield
                decoder_queue = [
                    batch
                    for batch in decoder_queue
                    if not all(task.check_dep() for task in batch)
                ]
                batch = encoder()
                if batch is not None:
                    decoder_queue.append(batch)
                yield

        tee_event = tee_loop()
        start_time = time.time()

        while self.finished_size < self.load_size:
            # disaggregate request
            reqs = self.request_source.arrive_requests()
            if len(reqs) > 0:
                add_metric("arrived_request_size", len(reqs))
            for req in reqs:
                pub, pri = Task.create_task_from_request(req)
                if pub is not None:
                    self.gpu_encoder_task_queue.append(pub)
                if pri is not None:
                    self.tee_encoder_task_queue.append(pri)
            gpu()
            next(tee_event)
            gpu_post()
