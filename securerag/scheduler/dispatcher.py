import time
from concurrent.futures import ThreadPoolExecutor, thread
from threading import Lock
from urllib import request

import grpc
import psutil
import torch

from securerag.data import Profiler
from securerag.rpc import messages_pb2_grpc
from securerag.scheduler.requests import RequestSource
from securerag.scheduler.tasks import (
    BatchDecoderTask,
    BatchEncoderTask,
    BatchGenerateTask,
    LocalEncoderDecoderService,
    Task,
    TaskQueue,
)
from securerag.utils import add_metric, get_metric, show_metric


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
        self.gpu_encoder_batch_size = 8
        self.gpu_decoder_batch_size = 8

    def registry_request_source(self, source: RequestSource):
        self.request_source = source

    def native_endpoint_loop(self, enable_offloading=True):
        task_queue = TaskQueue()
        batch_size = 1
        if enable_offloading:
            port = 8083
        else:
            port = 8082
        channel = grpc.insecure_channel(
            f"localhost:{port}",
            options=[
                ("grpc.max_receive_message_length", -1),
                ("grpc.max_send_message_length", -1),
            ],
        )
        stub = messages_pb2_grpc.GenerateServiceStub(channel)
        if self.request_source is None:
            print("self.request_source must be registred")
            return
        while self.finished_size < self.load_size:
            reqs = self.request_source.arrive_requests()
            task_queue += [Task.create_native_task_from_request(req) for req in reqs]
            batch = task_queue[:batch_size]
            for task in batch:
                task_queue.remove(task)
            if len(batch) == 0:
                continue
            batch_task = BatchGenerateTask().add_tasks(batch)
            batch_task.rpc_execute(stub, enable_offloading)
            batch_task.future.result()
            for task in batch_task.post_process():
                arrive_time = task.request.arrive_time
                finish_time = task.request.finish_time = time.time()
                add_metric("LATENCY", finish_time - arrive_time)
                add_metric("finished_request", 1)
            show_metric()

    def endpoint_loop_thread(self, service: LocalEncoderDecoderService):
        # cpus = range(0, 10, 1)
        # print(cpus, len(cpus))
        # psutil.Process().cpu_affinity(cpus)
        # print(psutil.Process().cpu_affinity())
        # torch.set_num_threads(len(cpus))

        if self.request_source is None:
            print("self.request_source must be registred")
            return
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.service = service

        def gpu():
            batch = self.gpu_encoder_task_queue[: self.gpu_encoder_batch_size]
            self.gpu_encoder_task_queue[:] = self.gpu_encoder_task_queue[len(batch) :]
            if len(batch) == 0:
                return
            service.ExecuteBatchEncoderTask(batch, device="cuda")
            batch = [Task.create_decoder_task(task, new_task=False) for task in batch]
            service.ExecuteBatchDecoderTask(batch, device="cuda")
            for task in batch:
                arrive_time = task.request.arrive_time
                finish_time = task.request.finish_time = time.time()
                task.is_finished = True
                print(
                    f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
                )
                add_metric("LATENCY_GPU", finish_time - arrive_time)
                add_metric("finished_request_gpu", 1)
            show_metric()
            return

        def tee_loop():
            while True:
                batch_passages_size = 1
                batch = []
                while len(batch) == 0:
                    batch = self.tee_encoder_task_queue.pop_encoder_tasks(
                        batch_passages_size
                    )
                    yield
                service.ExecuteBatchEncoderTask(batch)
                for task in batch:
                    task.is_finished = True
                    arrive_time = task.request.arrive_time
                    add_metric("LATENCY_ENCODER_TEE", time.time() - arrive_time)
                    add_metric("finished_encoder_tee", 1)
                with Profiler("SYNC_TEE"):
                    while not all(task.dep.resolve() for task in batch):
                        yield
                batch = [Task.create_decoder_task(task) for task in batch]
                service.ExecuteBatchDecoderTask(batch)
                for task in batch:
                    arrive_time = task.request.arrive_time
                    finish_time = task.request.finish_time = time.time()
                    task.is_finished = True
                    print(
                        f"question: {task.input.get('question')}, answer: {task.output.get('text_ans')}"
                    )
                    add_metric("LATENCY_TEE", finish_time - arrive_time)
                    add_metric("finished_request_tee", 1)
                    if len(get_metric("finished_request_tee")) == 55:
                        add_metric("FINISH_TIME_TEE", time.time() - start_time)
                show_metric()
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

    def endpoint_loop(self):
        if self.request_source is None:
            print("self.request_source must be registred")
            return
        tee_event = self.tee_loop()
        gpu_event = self.gpu_loop()

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
            # submit batch task
            next(gpu_event)
            next(tee_event)
            # collect result and check status
            for batch in self.rpc_execute_batch_task_queue:
                result = batch.post_process()
                if result is None:
                    continue
                self.rpc_execute_batch_task_queue.remove(batch)
                if type(batch) is BatchEncoderTask:
                    tasks = result
                    for task in tasks:
                        task = Task.create_decoder_task(task)
                        if task.env_type == "TEE":
                            self.tee_decoder_task_queue.append(task)
                        elif task.env_type == "GPU":
                            self.gpu_decoder_task_queue.append(task)
                elif type(batch) is BatchDecoderTask:
                    tasks = result
                    for task in tasks:
                        arrive_time = task.request.arrive_time
                        finish_time = task.request.finish_time = time.time()
                        if task.env_type == "GPU":
                            add_metric("LATENCY_GPU", finish_time - arrive_time)
                            add_metric("finished_request_gpu", 1)
                        elif task.env_type == "TEE":
                            add_metric("LATENCY_TEE", finish_time - arrive_time)
                            add_metric("finished_request_tee", 1)
                show_metric()
                print(
                    f"te: {len(self.tee_encoder_task_queue)}, td: {len(self.tee_decoder_task_queue)}, ge: {len(self.gpu_encoder_task_queue)}, gd: {len(self.gpu_decoder_task_queue)}, rpc: {len(self.rpc_execute_batch_task_queue)}"
                )

    def tee_loop(self):
        port = self.tee_service_port
        channel = grpc.insecure_channel(
            f"localhost:{port}",
            options=[
                ("grpc.max_receive_message_length", -1),
                ("grpc.max_send_message_length", -1),
            ],
        )
        encoder_stub = messages_pb2_grpc.EncoderServiceStub(channel)
        decoder_stub = messages_pb2_grpc.DecoderServiceStub(channel)

        def do_tee_loop():
            # disaggregate iteration
            encoder_task_waiting_time = self.tee_encoder_task_queue.total_waiting_time()
            decoder_task_waiting_time = self.tee_decoder_task_queue.total_waiting_time()

            # if encoder_task_waiting_time >= decoder_task_waiting_time:
            if decoder_task_waiting_time == 0:
                # schedule request priority
                tasks = self.tee_encoder_task_queue.pop_earliest_tasks(
                    self.tee_encoder_batch_size
                )
                if len(tasks) == 0:
                    return
                batch = BatchEncoderTask().add_tasks(tasks)
                batch.rpc_execute(encoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)
            else:
                # schedule request priority
                tasks = self.tee_decoder_task_queue.pop_earliest_tasks(
                    self.tee_decoder_batch_size, need_dependency=True
                )
                if len(tasks) == 0:
                    return
                batch = BatchDecoderTask().add_tasks(tasks)
                batch.rpc_execute(decoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)

        while self.running:
            do_tee_loop()
            yield

    def gpu_loop(self):
        port = self.gpu_service_port
        channel = grpc.insecure_channel(
            f"localhost:{port}",
            options=[
                ("grpc.max_receive_message_length", -1),
                ("grpc.max_send_message_length", -1),
            ],
        )
        encoder_stub = messages_pb2_grpc.EncoderServiceStub(channel)
        decoder_stub = messages_pb2_grpc.DecoderServiceStub(channel)

        def do_gpu_loop():
            # disaggregate iteration
            encoder_task_waiting_time = self.gpu_encoder_task_queue.total_waiting_time()
            decoder_task_waiting_time = self.gpu_decoder_task_queue.total_waiting_time()

            # if encoder_task_waiting_time >= decoder_task_waiting_time:
            if decoder_task_waiting_time == 0:
                # schedule request priority
                tasks = self.gpu_encoder_task_queue.pop_earliest_tasks(
                    self.gpu_encoder_batch_size
                )
                if len(tasks) == 0:
                    return
                batch = BatchEncoderTask().add_tasks(tasks)
                batch.rpc_execute(encoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)
            else:
                # schedule request priority
                tasks = self.gpu_decoder_task_queue.pop_earliest_tasks(
                    self.gpu_decoder_batch_size, need_dependency=True
                )
                if len(tasks) == 0:
                    return
                batch = BatchDecoderTask().add_tasks(tasks)
                batch.rpc_execute(decoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)

        while self.running:
            do_gpu_loop()
            yield
            yield
