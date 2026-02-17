import threading
from concurrent.futures import thread

import grpc

from securerag.rpc import messages_pb2_grpc
from securerag.scheduler.requests import RequestSource
from securerag.scheduler.tasks import (
    BatchDecoderTask,
    BatchEncoderTask,
    Task,
    TaskQueue,
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

        self.tee_encoder_batch_size = 2
        self.tee_decoder_batch_size = 2
        self.gpu_encoder_batch_size = 16
        self.gpu_decoder_batch_size = 16

    def registry_request_source(self, source: RequestSource):
        self.request_source = source

    def endpoint_loop(self):
        if self.request_source is None:
            print("self.request_source must be registred")
            return
        tee_event = self.tee_loop()
        gpu_event = self.gpu_loop()

        while self.finished_size < self.load_size:
            # disaggregate request
            reqs = self.request_source.arrive_requests()
            print("endpoint-loop")
            for req in reqs:
                pub, pri = Task.create_task_from_request(req)
                if pub is not None:
                    self.tee_encoder_task_queue.append(pub)
                if pri is not None:
                    self.gpu_encoder_task_queue.append(pri)
            next(tee_event)
            next(gpu_event)
            # status check
            for batch in self.rpc_execute_batch_task_queue:
                result = batch.post_process()
                if result is None:
                    continue
                if type(batch) is BatchEncoderTask:
                    tasks = result
                    for task in tasks:
                        if task.dep is not None and task.dep.resolve() is False:
                            continue
                        task = Task.create_decoder_task(task)
                        if task.env_type == "TEE":
                            self.tee_decoder_task_queue.append(task)
                        elif task.env_type == "GPU":
                            self.gpu_decoder_task_queue.append(task)
                elif type(batch) is BatchDecoderTask:
                    pass

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
            print("tee-loop")
            # disaggregate iteration
            encoder_task_waiting_time = self.tee_encoder_task_queue.total_waiting_time()
            decoder_task_waiting_time = self.tee_decoder_task_queue.total_waiting_time()

            if encoder_task_waiting_time >= decoder_task_waiting_time:
                # schedule request priority
                tasks = self.tee_encoder_task_queue.pop_shortest_tasks(
                    self.tee_encoder_batch_size
                )
                if len(tasks) == 0:
                    return
                batch = BatchEncoderTask().add_tasks(tasks)
                batch.rpc_execute(encoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)
            else:
                # schedule request priority
                tasks = self.tee_decoder_task_queue.pop_shortest_tasks(
                    self.tee_decoder_batch_size
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
            print("gpu-loop")
            # disaggregate iteration
            encoder_task_waiting_time = self.gpu_encoder_task_queue.total_waiting_time()
            decoder_task_waiting_time = self.gpu_decoder_task_queue.total_waiting_time()

            if encoder_task_waiting_time >= decoder_task_waiting_time:
                # schedule request priority
                tasks = self.gpu_encoder_task_queue.pop_shortest_tasks(
                    self.gpu_encoder_batch_size
                )
                if len(tasks) == 0:
                    return
                batch = BatchEncoderTask().add_tasks(tasks)
                batch.rpc_execute(encoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)
            else:
                # schedule request priority
                tasks = self.gpu_decoder_task_queue.pop_shortest_tasks(
                    self.gpu_decoder_batch_size
                )
                if len(tasks) == 0:
                    return
                batch = BatchDecoderTask().add_tasks(tasks)
                batch.rpc_execute(decoder_stub)
                self.rpc_execute_batch_task_queue.append(batch)

        while self.running:
            do_gpu_loop()
            yield
